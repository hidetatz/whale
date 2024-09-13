package main

/*
	#cgo LDFLAGS: -ldl
	#include <dlfcn.h>
	#include <stdlib.h>

	void setFloatArray(float *arr, float *values, int size) {
      for (int i = 0; i < size; i++) { arr[i] = values[i]; }
    }

	float* callFunc(void *p, char** err) {
	  float* (*f)() = p;
	  return f(err);
	}
*/
import "C"

import (
	"fmt"
	"os"
	"os/exec"
	"strings"
	"unsafe"
)

type cuda struct {
	indentlevel int
}

func (c *cuda) f(format string, a ...any) string {
	return c.indent() + fmt.Sprintf(format, a...)
}

func (c *cuda) dataOnHost(idx, length int) []string {
	return []string{
		c.f("float data_host_%v[%v];", idx, length),
	}
}

func (c *cuda) memcpyHostToDevice(idx int) []string {
	return []string{
		c.f("float *data%v;", idx),
		c.f("CHECKERR(cudaMalloc((float**)&data%v, nbytes));", idx),
		c.f("CHECKERR(cudaMemcpy(data%v, data_host_%v, nbytes, cudaMemcpyHostToDevice));", idx, idx),
	}
}

func (c *cuda) alu2(fn string, result, left, right int) []string {
	return []string{
		c.f("float *data%v;", result),
		c.f("CHECKERR(cudaMalloc((float**)&data%v, nbytes));", result),
		c.f("%v<<<grid, block>>>(data%v, data%v, data%v);", fn, left, right, result),
	}
}

func (c *cuda) returnresult(idx int) []string {
	return []string{
		c.f("float *result = (float *)malloc(nbytes);"),
		c.f("memset(result, 0, nbytes);"),
		c.f("CHECKERR(cudaMemcpy(result, data%v, nbytes, cudaMemcpyDeviceToHost));", idx),
		c.f("cudaDeviceReset();"),
		c.f("return result;"),
	}
}

func (c *cuda) indent() string {
	return strings.Repeat("  ", c.indentlevel)
}

func (c *cuda) run(tasks []*task) []float32 {
	resultlen := len(tasks[0].constant) // todo: calculate properly

	computes := [][]string{}
	inputIdx := []int{}
	inputCPU := [][]string{}
	inputGPU := [][]string{}
	var returnresult []string

	for i, task := range tasks {
		switch task.op {

		case ops.constant:
			inputIdx = append(inputIdx, i)
			inputCPU = append(inputCPU, c.dataOnHost(i, len(tasks[i].constant)))
			inputGPU = append(inputGPU, c.memcpyHostToDevice(i))

		case ops.add:
			computes = append(computes, c.alu2("add", i, task.inputs[0], task.inputs[1]))

		case ops.mul:
			computes = append(computes, c.alu2("mul", i, task.inputs[0], task.inputs[1]))
		}

		if i == len(tasks)-1 {
			returnresult = c.returnresult(i)
		}
	}

	prg := `#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mul(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] * B[i];
}

__global__ void add(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

// inputsCPU
%s

#define CHECKERR(call) \
{\
  const cudaError_t err = call;\
  if (err != cudaSuccess) {\
    *cerr = (char*)cudaGetErrorString(err);\
  }\
}

extern "C" float* f(char** cerr) {
  int dev = 0;  
  cudaSetDevice(dev);

  int n = %d; // vector size
  size_t nbytes = n * sizeof(float);

  // inputsGPU
  %s

  dim3 grid(1);
  dim3 block(n);

  // computes
  %s

  // return result
  %s
}
`

	f, err := os.Create("/tmp/f.cu")
	if err != nil {
		panic(err)
	}

	join1 := func(ss []string, sep string) string {
		return strings.Join(ss, sep)
	}

	join2 := func(sss [][]string, sep2, sep1 string) string {
		s := make([]string, len(sss))
		for i := range sss {
			s[i] = join1(sss[i], sep1)
		}
		return strings.Join(s, sep2)
	}

	program := fmt.Sprintf(prg,
		join2(inputCPU, "\n", "\n"),
		resultlen,
		join2(inputGPU, "\n\n  ", "\n  "),
		join2(computes, "\n\n  ", "\n  "),
		join1(returnresult, "\n  "),
	)

	f.WriteString(program)
	f.Close()

	out, err := exec.Command("nvcc", "-O3", "--shared", "-Xcompiler", "-fPIC", "-o", "/tmp/f.so", "/tmp/f.cu").CombinedOutput()
	if err != nil {
		panic(string(out))
	}

	soname := C.CString("/tmp/f.so")
	defer C.free(unsafe.Pointer(soname))

	handle := C.dlopen(soname, C.RTLD_LAZY)
	if handle == nil {
		panic("dlopen fail: " + C.GoString(C.dlerror()))
	}

	defer C.dlclose(handle)

	for _, idx := range inputIdx {
		dataname := C.CString(fmt.Sprintf("data_host_%v", idx))
		defer C.free(unsafe.Pointer(dataname))
		d := C.dlsym(handle, dataname)
		if d == nil {
			panic(fmt.Sprintf("error resolving d: %v", C.GoString(C.dlerror())))
		}

		C.setFloatArray((*C.float)(d), (*C.float)(&tasks[idx].constant[0]), C.int(len(tasks[idx].constant)))
	}

	fname := C.CString("f")
	defer C.free(unsafe.Pointer(fname))
	fp := C.dlsym(handle, fname)
	if fp == nil {
		panic(fmt.Sprintf("error resolving f: %v", C.GoString(C.dlerror())))
	}

	var cerr *C.char
	resultp := C.callFunc(fp, &cerr)
	if resultp == nil {
		panic(fmt.Sprintf("error on f() : %v", C.GoString(C.dlerror())))
	}

	result := (*[1 << 30]float32)(unsafe.Pointer(resultp))[:resultlen:resultlen]

	res := make([]float32, len(result))
	for i, v := range result {
		res[i] = v
	}

	return res
}

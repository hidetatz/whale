package main

import (
	// #cgo LDFLAGS: -ldl
	// #include <stdlib.h>
	// #include <dlfcn.h>
	// float* call_c_func(void *p) {
	//   float* (*f)() = p;
	//   return f();
	// }
	"C"

	"fmt"
	"os"
	"os/exec"
	"plugin"
	"strings"
	"unsafe"
)

type op int

type _ops struct {
	constant op
	add      op
}

// Pseudo-namespacing
var ops = &_ops{
	1, 2,
}

type backend interface {
	run(tasks []*task) []float32
}

var runner backend

func init() {
	available := func(cmd string) bool {
		err := exec.Command("bash", "-c", "command -v "+cmd).Run()
		return err == nil // likely the cmd is available
	}

	if available("nvcc") {
		runner = &cuda{}
		return
	}

	runner = &golang{}
}

type cuda struct{}

func (_ *cuda) run(tasks []*task) []float32 {
	resultlen := len(tasks[0].data)
	prgs := []string{}
	params := []int{}
	for i, task := range tasks {
		switch task.op {
		case ops.constant:
			params = append(params, i)
		case ops.add:
			prgs = append(prgs, fmt.Sprintf(`
  float *data%v;
  checkerr(cudaMalloc((float**)&data%v, nbytes), %v);
  add<<<grid, block>>>(data%v, data%v, data%v);
`, i, i, i, task.inputs[0], task.inputs[1], i))
		}

		if i == len(tasks)-1 {
			prgs = append(prgs, fmt.Sprintf("  float *result = (float *)malloc(nbytes);"))
			prgs = append(prgs, fmt.Sprintf("  memset(result, 0, nbytes);"))
			prgs = append(prgs, fmt.Sprintf("  checkerr(cudaMemcpy(result, data%v, nbytes, cudaMemcpyDeviceToHost), %v);", i, i))
			prgs = append(prgs, fmt.Sprintf("  return result;"))
		}
	}

	sparams := make([]string, len(params))
	sparamsGPU := make([]string, len(params))
	vals := func(fs []float32) string {
		s := ""
		for i := range fs {
			s += fmt.Sprintf("%f", fs[i])

			if i != len(fs)-1 {
				s += ", "
			}
		}
		return s
	}
	for i := range params {
		sparams[i] = fmt.Sprintf(`float data_host_%v[] = {%s};`, params[i], vals(tasks[params[i]].data))

		sparamsGPU[i] = fmt.Sprintf(`
  float *data%v;
  checkerr(cudaMalloc((float**)&data%v, nbytes), %v);
  checkerr(cudaMemcpy(data%v, data_host_%v, nbytes, cudaMemcpyHostToDevice), %v);
`, params[i], params[i], params[i], params[i], params[i], params[i])
	}

	prg := `#include <cuda_runtime.h>
#include <stdio.h>

__global__ void add(float *A, float *B, float *C) {
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}

// data on host definitions
%s

void checkerr(cudaError_t err, int i) {
  if (err != cudaSuccess) {
    printf("cuda error (%%d): %%s\n", i, cudaGetErrorString(err));
    exit(1);
  }
}

extern "C" float* f() {
  int dev = 0;  
  cudaSetDevice(dev);

  int n = %d; // vector size
  size_t nbytes = n * sizeof(float);

  // load on GPU
%s

  dim3 grid(1);
  dim3 block(n);

  // computation
%s

  cudaDeviceReset();
}
`

	f, err := os.Create("/tmp/f.cu")
	if err != nil {
		panic(err)
	}

	program := fmt.Sprintf(prg, strings.Join(sparams, "\n"), len(tasks[0].data), strings.Join(sparamsGPU, "\n"), strings.Join(prgs, "\n"))

	f.WriteString(program)
	f.Close()

	out, err := exec.Command("nvcc", "--shared", "-Xcompiler", "-fPIC", "-o", "/tmp/f.so", "/tmp/f.cu").CombinedOutput()
	if err != nil {
		panic(string(out))
	}

	soname := C.CString("/tmp/f.so")
	defer C.free(unsafe.Pointer(soname))

	handle := C.dlopen(soname, C.RTLD_LAZY)
	if handle == nil {
		panic("dlopen fail!")
	}

	C.dlerror()
	fnPointer := C.dlsym(handle, C.CString("f"))
	e := C.dlerror()
	if e != nil {
		panic(fmt.Sprintf("error resolving f: %v", C.GoString(e)))
	}

	p := C.call_c_func(fnPointer)
	defer C.free(unsafe.Pointer(p))

	result := make([]float32, resultlen)
	for i := range result {
		result[i] = float32(*(*float32)(unsafe.Pointer(uintptr(unsafe.Pointer(p)) + uintptr(i)*4)))
	}

	return result
}

type golang struct{}

func (_ *golang) run(tasks []*task) []float32 {
	prgs := []string{}
	params := []int{}
	for i, task := range tasks {
		switch task.op {
		case ops.constant:
			params = append(params, i)
		case ops.add:
			prgs = append(prgs, fmt.Sprintf("	Data%v := add(Data%v, Data%v)", i, task.inputs[0], task.inputs[1]))
		}

		if i == len(tasks)-1 {
			prgs = append(prgs, fmt.Sprintf("	Result = Data%v", i))
		}
	}

	prg := `package main

%v

var Result []float32

func add(a []float32, b []float32) []float32 {
	data := make([]float32, len(a))
	for i := range a {
		data[i] = a[i] + b[i]
	}
	return data
}

func F() {
%v
}
`

	sparams := make([]string, len(params))
	for i := range params {
		sparams[i] = fmt.Sprintf("var Data%v []float32", params[i])
	}

	f, err := os.Create("/tmp/f.go")
	if err != nil {
		panic(err)
	}

	program := fmt.Sprintf(prg, strings.Join(sparams, "\n"), strings.Join(prgs, "\n"))

	f.WriteString(program)
	f.Close()

	out, err := exec.Command("go", "build", "-o", "/tmp/f.so", "-buildmode=plugin", "/tmp/f.go").CombinedOutput()
	if err != nil {
		panic(string(out))
	}

	p, err := plugin.Open("/tmp/f.so")
	if err != nil {
		panic(err)
	}

	for _, param := range params {
		d, err := p.Lookup(fmt.Sprintf("Data%v", param))
		if err != nil {
			panic(err)
		}

		*d.(*[]float32) = tasks[param].data
	}

	fn, err := p.Lookup("F")
	if err != nil {
		panic(err)
	}

	fn.(func())()

	result, err := p.Lookup("Result")
	if err != nil {
		panic(err)
	}

	return *result.(*[]float32)
}

type Tensor struct {
	op   op
	src  []*Tensor
	data []float32
}

func Empty(op op) *Tensor {
	return &Tensor{op: op}
}

func New(data []float32) *Tensor {
	return &Tensor{op: ops.constant, data: data}
}

type task struct {
	op     op
	data   []float32
	inputs []int
}

func (t *Tensor) toposort() []*task {
	// todo: do toposort
	task0 := &task{op: t.src[0].src[0].op, data: t.src[0].src[0].data}
	task1 := &task{op: t.src[0].src[1].op, data: t.src[0].src[1].data}
	task2 := &task{op: t.src[0].op, inputs: []int{0, 1}}
	task3 := &task{op: t.src[1].op, data: t.src[1].data}
	task4 := &task{op: t.op, inputs: []int{2, 3}}
	return []*task{task0, task1, task2, task3, task4}
}

func (t *Tensor) Materialize() []float32 {
	tasks := t.toposort()
	result := runner.run(tasks)
	t.data = result
	return t.data
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	y := Empty(ops.add)
	y.src = []*Tensor{t, t2}
	return y
}

func main() {
	t := New([]float32{1, 2})
	t2 := New([]float32{3, 4})
	t3 := t.Add(t2)
	t4 := t3.Add(New([]float32{10, 10}))

	t4.Materialize()
	fmt.Println(t4.data)
}

package main

import (
	/*
	   #cgo LDFLAGS: -L/opt/OpenBLAS/lib/ -lopenblas
	   #cgo CFLAGS: -I /opt/OpenBLAS/include/
	   #include <cblas.h>
	*/
	"C"
	"fmt"
	"unsafe"
)
import (
	"github.com/hidetatz/whale/tensor"
)

func main() {
	t1 := tensor.New([][]float32{
		{1, 2, 3},
		{4, 5, 6},
	})

	t2 := tensor.New([][]float32{
		{1, 2, 3},
		{4, 5, 6},
		{7, 8, 9},
	})

	m := C.blasint(t1.Shape[0])
	k := C.blasint(t1.Shape[1])
	n := C.blasint(t2.Shape[1])

	alpha := C.float(1)
	beta := C.float(0)

	c := make([]float32, t1.Shape[0]*t2.Shape[1])

	// void cblas_sgemm(
	// 	OPENBLAS_CONST enum CBLAS_ORDER Order,
	// 	OPENBLAS_CONST enum CBLAS_TRANSPOSE TransA,
	// 	OPENBLAS_CONST enum CBLAS_TRANSPOSE TransB,
	// 	OPENBLAS_CONST blasint M,
	// 	OPENBLAS_CONST blasint N,
	// 	OPENBLAS_CONST blasint K,
	// 	OPENBLAS_CONST float alpha,
	// 	OPENBLAS_CONST float *A,
	// 	OPENBLAS_CONST blasint lda,
	// 	OPENBLAS_CONST float *B,
	// 	OPENBLAS_CONST blasint ldb,
	// 	OPENBLAS_CONST float beta,
	// 	float *C,
	// 	OPENBLAS_CONST blasint ldc
	// );
	C.cblas_sgemm(
		C.CblasRowMajor,
		C.CblasNoTrans,
		C.CblasNoTrans,
		m,
		n,
		k,
		alpha,
		(*C.float)(unsafe.Pointer(&t1.Ravel()[0])),
		k,
		(*C.float)(unsafe.Pointer(&t2.Ravel()[0])),
		n,
		beta,
		(*C.float)(unsafe.Pointer(&c[0])),
		n,
	)

	fmt.Println(c)
}

// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib

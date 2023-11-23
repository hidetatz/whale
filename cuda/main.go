package main

/*
void maxmul(float *A, float* B, float *C, int size);
#cgo LDFLAGS: -L. -L./ -lmaxmul
*/
import "C"

import "fmt"

func Maxmul(a []C.float, b []C.float, c []C.float, size int) {
	C.maxmul(&a[0], &b[0], &c[0], C.int(size))
}

func main() {
	//in := []C.float{1.23, 4.56}
	//C.test(&in[0]) // C 1.230000 4.560000
	a := []C.float{-1, 2, 4, 0, 5, 3, 6, 2, 1}
	b := []C.float{3, 0, 2, 3, 4, 5, 4, 7, 2}
	var c []C.float = make([]C.float, 9)
	Maxmul(a, b, c, 3)
	fmt.Println(c)
}

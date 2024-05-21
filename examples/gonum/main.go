package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func main() {
	A := mat.NewDense(3, 4, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	B := mat.NewDense(4, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})

	C := mat.NewDense(3, 3, nil)
	C.Product(A, B)
	fmt.Println(C)

}

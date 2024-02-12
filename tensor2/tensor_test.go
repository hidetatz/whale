package tensor2

import (
	"fmt"
	"testing"
)

func TestString(t *testing.T) {
	tensor := MustNd(seqf(1, 7), 2, 3)
	fmt.Println(tensor)

	tensor2, err := tensor.Index(0, 2)
	checkErr(t, false, err)
	fmt.Println(tensor2)

	tensor3 := MustNd(seqf(1, 25), 2, 3, 4)
	fmt.Println(tensor3)

	tensor4, err := tensor3.Slice(From(1), To(1), FromTo(1, 3))
	checkErr(t, false, err)
	fmt.Println(tensor4)

	tensor5, err := tensor4.Slice(From(1), To(1), FromTo(1, 2))
	checkErr(t, false, err)
	fmt.Println(tensor5)
}

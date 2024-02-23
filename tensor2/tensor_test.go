package tensor2

import (
	"fmt"
	"testing"
)

func TestString(t *testing.T) {
	musteq := func(t *testing.T, got, expected string) {
		t.Helper()
		if got != expected {
			t.Errorf("String() mismatch: got: '%v', expected: '%v'", got, expected)
		}
	}

	tensor := MustNd(seqf(1, 7), 2, 3)
	got := fmt.Sprintf("%v", tensor)
	musteq(t, got, `[
  [1, 2, 3]
  [4, 5, 6]
]
`)

	tensor2, err := tensor.Index(0, 2)
	checkErr(t, false, err)
	got = fmt.Sprintf("%v", tensor2)
	musteq(t, got, `3`)

	tensor3 := MustNd(seqf(1, 25), 2, 3, 4)
	got = fmt.Sprintf("%v", tensor3)
	musteq(t, got, `[
  [
    [1, 2, 3, 4]
    [5, 6, 7, 8]
    [9, 10, 11, 12]
  ]
  [
    [13, 14, 15, 16]
    [17, 18, 19, 20]
    [21, 22, 23, 24]
  ]
]
`)

	tensor4, err := tensor3.Slice(From(1), To(1), FromTo(1, 3))
	checkErr(t, false, err)
	got = fmt.Sprintf("%v", tensor4)
	musteq(t, got, `[
  [
    [14, 15]
  ]
]
`)

	tensor5, err := tensor4.Slice(From(1), To(1), FromTo(1, 2))
	checkErr(t, false, err)
	got = fmt.Sprintf("%v", tensor5)
	// mimics numpy
	musteq(t, got, `([], shape=[0 1 1])`)
}

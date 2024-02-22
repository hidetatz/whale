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
  [1.00, 2.00, 3.00]
  [4.00, 5.00, 6.00]
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
    [1.00, 2.00, 3.00, 4.00]
    [5.00, 6.00, 7.00, 8.00]
    [9.00, 10.00, 11.00, 12.00]
  ]
  [
    [13.00, 14.00, 15.00, 16.00]
    [17.00, 18.00, 19.00, 20.00]
    [21.00, 22.00, 23.00, 24.00]
  ]
]
`)

	tensor4, err := tensor3.Slice(From(1), To(1), FromTo(1, 3))
	checkErr(t, false, err)
	got = fmt.Sprintf("%v", tensor4)
	musteq(t, got, `[
  [
    [14.00, 15.00]
  ]
]
`)

	tensor5, err := tensor4.Slice(From(1), To(1), FromTo(1, 2))
	checkErr(t, false, err)
	got = fmt.Sprintf("%v", tensor5)
	// mimics numpy
	musteq(t, got, `([], shape=[0 1 1])`)
}

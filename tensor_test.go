package whale

import (
	"fmt"
	"testing"
)

func TestTensorString(t *testing.T) {
	ts, _ := TensorNd([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 8)
	fmt.Println(ts)

	ts, _ = TensorNd([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 1, 2, 1, 2, 2)
	fmt.Println(ts)
}

// func TestTensorString(t *testing.T) {
// 	ts := TensorNd([]float64{1, 2}, 2)
// 	got, err := ts.BroadcastTo(2, 2, 2)
// 	if err != nil {
// 		t.Errorf("unexpected error: %v", err)
// 	}
//
// 	expected := TensorNd([]float64{1, 2, 1, 2, 1, 2, 1, 2}, 2, 2, 2)
// 	if !ts.Equals(expected) {
// 		t.Errorf("expected: '%s', got: '%s'", expected, got)
// 	}
// }
//
// func TestTensor_String(t *testing.T) {
// 	ts := Ones(2, 3, 3)
//
// 	expected := `[
//   [
//     [1 1 1]
//   ]
//   [
//     [1 1 1]
//   ]
//   [
//     [1 1 1]
//   ]
// ]
// [
//   [
//     [1 1 1]
//   ]
//   [
//     [1 1 1]
//   ]
//   [
//     [1 1 1]
//   ]
// ]
// `
//
// 	if ts.String() != expected {
// 		t.Errorf("expected: '%s', got: '%s'", expected, ts.String())
// 	}
//
// 	ts = TensorNd([]float64{1, 2}, 1, 1, 2)
//
// 	expected = `[
//   [
//     [1 2]
//   ]
// ]
// `
//
// 	if ts.String() != expected {
// 		t.Errorf("expected: '%s', got: '%s'", expected, ts.String())
// 	}
// }

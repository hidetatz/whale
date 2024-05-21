package tensor

import (
	"fmt"
	"reflect"
	"slices"
	"testing"
)

func TestTensor(t *testing.T) {
	assert := func(t *testing.T, expected, got any) {
		t.Helper()
		if !reflect.DeepEqual(expected, got) {
			t.Fatalf("expected: %v, got: %v", expected, got)
		}
	}

	sc := Scalar(3)

	assert(t, 0, sc.Ndim())
	assert(t, 1, sc.Size())
	assert(t, true, sc.IsScalar())
	assert(t, float32(3.0), sc.AsScalar())
	assert(t, false, sc.IsVector())

	vc := Vector([]float32{1, 2, 3})
	assert(t, 1, vc.Ndim())
	assert(t, 3, vc.Size())
	assert(t, false, vc.IsScalar())
	assert(t, true, vc.IsVector())
	assert(t, []float32{1, 2, 3}, vc.AsVector())

	t1 := Arange(0, 20, 1).Reshape(4, 5)
	assert(t, 2, t1.Ndim())
	assert(t, 20, t1.Size())
	assert(t, false, t1.IsScalar())
	assert(t, false, t1.IsVector())
}

func TestEquals(t *testing.T) {
	assert := func(t *testing.T, b bool) {
		t.Helper()
		if !b {
			t.Fatalf("expected true")
		}
	}

	t1 := Vector([]float32{0, 1, 2, 1, 0})
	t2 := New([][]float32{
		{0, 1, 2, 1, 0},
		{1, 1, 1, 1, 1},
		{2, 2, 2, 2, 2},
		{1, 1, 1, 1, 1},
		{0, 1, 2, 1, 0},
	})

	assert(t, t1.Equals(New([]float32{0, 1, 2, 1, 0})))
	assert(t, t1.Equals(t2.Index(At(0))))
	assert(t, t1.Equals(t2.Index(At(4))))
	assert(t, t1.Equals(t2.Index(All(), At(0))))
	assert(t, t1.Equals(t2.Index(All(), At(4))))
}

func TestFlatten(t *testing.T) {
	tests := []struct {
		name     string
		tensor   *Tensor
		expected []float32
	}{
		{
			name:     "scalar",
			tensor:   Scalar(3),
			expected: []float32{3},
		},
		{
			name:     "vector",
			tensor:   Vector([]float32{0, 1, 2, 3}),
			expected: []float32{0, 1, 2, 3},
		},
		{
			name:     "2d",
			tensor:   Arange(0, 4, 1).Reshape(2, 2),
			expected: []float32{0, 1, 2, 3},
		},
		{
			name:     "2d indexed",
			tensor:   Arange(0, 4, 1).Reshape(2, 2).Index(At(1)),
			expected: []float32{2, 3},
		},
		{
			name:     "2d transposed",
			tensor:   Arange(0, 4, 1).Reshape(2, 2).Transpose(),
			expected: []float32{0, 2, 1, 3},
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := tc.tensor.Flatten()
			if !slices.Equal(got, tc.expected) {
				t.Fatalf("expected: %v, got: %v", tc.expected, got)
			}
		})
	}
}

func TestString(t *testing.T) {
	musteq := func(t *testing.T, got, expected string) {
		t.Helper()
		if got != expected {
			t.Errorf("String() mismatch: got: '%v', expected: '%v'", got, expected)
		}
	}

	tensor := Vector([]float32{1, 2, 3, 4})
	got := fmt.Sprintf("%v", tensor)
	musteq(t, got, `[1, 2, 3, 4]`)

	tensor = Arange(1, 7, 1).Reshape(2, 3)
	got = fmt.Sprintf("%v", tensor)
	musteq(t, got, `[
  [1, 2, 3]
  [4, 5, 6]
]
`)

	tensor2 := tensor.Index(At(0), At(2))
	got = fmt.Sprintf("%v", tensor2)
	musteq(t, got, `3`)

	tensor3 := Arange(1, 25, 1).Reshape(2, 3, 4)
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

	tensor4 := tensor3.Index(From(1), To(1), FromTo(1, 3))
	got = fmt.Sprintf("%v", tensor4)
	musteq(t, got, `[
  [
    [14, 15]
  ]
]
`)

	tensor5 := tensor4.Index(From(1), To(1), FromTo(1, 2))
	got = fmt.Sprintf("%v", tensor5)
	// mimics numpy
	musteq(t, got, `([], shape=[0 1 1])`)

	tensor = Arange(1, 7, 1).Reshape(2, 3)
	musteq(t, tensor.OnelineString(), `[[1, 2, 3], [4, 5, 6]]`)
	musteq(t, tensor.Raw(), "{data: [1 2 3 4 5 6], offset: 0, Shape: [2 3], Strides: [3 1], isview: true}")
}

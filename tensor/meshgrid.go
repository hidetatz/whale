package tensor

import "fmt"

func MeshGrid(t1, t2 *Tensor) (*Tensor, *Tensor) {
	return MustGet2(RespErr.MeshGrid(t1, t2))
}

func (_ *plainErrResponser) MeshGrid(t1, t2 *Tensor) (*Tensor, *Tensor, error) {
	if !t1.IsVector() || !t2.IsVector() {
		return nil, nil, fmt.Errorf("argument must be vector")
	}

	newshape := []int{t2.Size(), t1.Size()}
	v1, v2 := t1.Flatten(), t2.Flatten()

	var d1 []float32
	for _ = range t2.Size() {
		d1 = append(d1, v1...)
	}

	r1, err := RespErr.NdShape(d1, newshape...)
	if err != nil {
		return nil, nil, err
	}

	var d2 []float32
	for i := range v2 {
		d2 = append(d2, all(v2[i], t1.Size())...)
	}

	r2, err := RespErr.NdShape(d2, newshape...)
	if err != nil {
		return nil, nil, err
	}

	return r1, r2, nil
}

package whale

import (
	"fmt"
	"sort"

	"github.com/hidetatz/whale/tensor"
)

var EnableBackprop = true

type Variable struct {
	data       *tensor.Tensor
	grad       *Variable
	creator    *function
	generation int
}

func NewVar(data *tensor.Tensor) *Variable {
	return &Variable{data: data}
}

func (v *Variable) String() string {
	return fmt.Sprintf("%v", v.data)
}

func (v *Variable) clone() *Variable {
	g := *v.grad
	return &Variable{
		data:       v.data.Copy(),
		grad:       &g,
		creator:    v.creator,
		generation: v.generation,
	}
}

func (v *Variable) Len() int {
	return len(v.data.Data)
}

func (v *Variable) Sub(lr float64) {
	newData := device.Sub(v.data, device.Mul(v.grad.data, tensor.FromScalar(lr)))
	v.data = newData
}

func (v *Variable) SetData(t *tensor.Tensor) {
	v.data = t
}

func (v *Variable) GetData() *tensor.Tensor {
	return v.data
}

func (v *Variable) GetGrad() *Variable {
	return v.grad
}

func (v *Variable) ClearGrad() {
	v.grad = nil
}

func (v *Variable) SetCreator(creator *function) {
	v.creator = creator
	v.generation = creator.generation + 1
}

func (v *Variable) Backward() error {
	if v.grad == nil {
		v.grad = NewVar(tensor.OnesLike(v.data))
	}

	fs := []*function{}
	uniqueadd := func(f *function) {
		for _, added := range fs {
			if added == f {
				return
			}
		}
		fs = append(fs, f)
		sort.Slice(fs, func(i, j int) bool {
			return fs[i].generation < fs[j].generation
		})
	}

	uniqueadd(v.creator)

	for len(fs) > 0 {
		var last *function
		last, fs = fs[len(fs)-1], fs[:len(fs)-1] // pop last

		ys := []*Variable{}
		for _, o := range last.outputs {
			ys = append(ys, o.grad)
		}

		gxs, err := last.op.Backward(ys...)
		if err != nil {
			return err
		}

		for i, x := range last.inputs {
			gx := gxs[i]
			if x.grad == nil {
				x.grad = gx
			} else {
				xg, err := Add(x.grad, gx)
				if err != nil {
					return err
				}

				x.grad = xg
			}

			if x.creator != nil {
				uniqueadd(x.creator)
			}
		}
	}

	return nil
}

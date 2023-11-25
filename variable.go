package whale

import (
	"fmt"
	"sort"
)

var EnableBackprop = true

type Variable struct {
	data       float64
	grad       *float64
	creator    *function
	generation int
}

func NewVar(data float64) *Variable {
	return &Variable{data: data}
}

func (v *Variable) String() string {
	return fmt.Sprintf("%v", v.data)
}

func (v *Variable) clone() *Variable {
	g := *v.grad
	return &Variable{
		data:       v.data,
		grad:       &g,
		creator:    v.creator,
		generation: v.generation,
	}
}

func (v *Variable) ClearGrad() {
	v.grad = nil
}

func (v *Variable) SetCreator(creator *function) {
	v.creator = creator
	v.generation = creator.generation + 1
}

func (v *Variable) Backward() {
	if v.grad == nil {
		g := 1.0
		v.grad = &g
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

		ys := []float64{}
		for _, o := range last.outputs {
			ys = append(ys, *o.grad)
		}

		gxs := last.operation.Backward(ys...)
		for i, x := range last.inputs {
			gx := gxs[i]
			if x.grad == nil {
				x.grad = &gx
			} else {
				s := *x.grad + gx
				x.grad = &s
			}

			if x.creator != nil {
				uniqueadd(x.creator)
			}
		}

	}
}

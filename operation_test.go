package whale

import (
	"reflect"
	"slices"
	"testing"

	"github.com/hidetatz/whale/tensor"
)

// Tests each calculation function correctness.
// Detailed operation test should be done in tensor/device tests.
// This test should focus on the correctness of forward/backward implementation.
func TestSingleOperations(t *testing.T) {
	tests := []struct {
		name string

		fn any
		in []*tensor.Tensor
		// extra arguments to the fn
		extra []any
		// flag if the last param of arguments is variable length
		variable bool

		expected     []*tensor.Tensor
		expectedGrad []*tensor.Tensor
	}{
		{
			name:         "reshape",
			fn:           Reshape,
			in:           ts(arng(t, 1, 13, 2, 2, 3)),
			extra:        []any{[]int{6, 2}},
			variable:     true,
			expected:     ts(arng(t, 1, 13, 6, 2)),
			expectedGrad: ts(ones(t, 2, 2, 3)),
		},
		{
			name:         "transpose",
			fn:           Transpose,
			in:           ts(arng(t, 1, 13, 2, 2, 3)),
			expected:     ts(trans(t, arng(t, 1, 13, 2, 2, 3))),
			expectedGrad: ts(ones(t, 2, 2, 3)),
		},
		{
			name:         "broadcastto",
			fn:           BroadcastTo,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{[]int{3, 2, 3}},
			variable:     true,
			expected:     ts(nd(t, []float64{1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, 3, 2, 3)),
			expectedGrad: ts(nd(t, []float64{3, 3, 3, 3, 3, 3}, 2, 3)),
		},
		{
			name:         "sum",
			fn:           Sum,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{false, []int{0}},
			variable:     true,
			expected:     ts(nd(t, []float64{5, 7, 9}, 3)),
			expectedGrad: ts(ones(t, 2, 3)),
		},
		{
			name:         "sum2",
			fn:           Sum,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{false, []int{1}},
			variable:     true,
			expected:     ts(nd(t, []float64{6, 15}, 2)),
			expectedGrad: ts(ones(t, 2, 3)),
		},
		{
			name:         "sum3",
			fn:           Sum,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{true, []int{1}},
			variable:     true,
			expected:     ts(nd(t, []float64{6, 15}, 2, 1)),
			expectedGrad: ts(ones(t, 2, 3)),
		},
		{
			name:         "sumTo",
			fn:           SumTo,
			in:           ts(arng(t, 1, 7, 2, 3)),
			extra:        []any{[]int{1, 3}},
			variable:     true,
			expected:     ts(nd(t, []float64{5, 7, 9}, 1, 3)),
			expectedGrad: ts(ones(t, 2, 3)),
		},
		{
			name:         "exp",
			fn:           Exp,
			in:           ts(arng(t, 1, 7, 2, 3)),
			expected:     ts(nd(t, []float64{2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 148.4131591025766, 403.4287934927351}, 2, 3)),
			expectedGrad: ts(nd(t, []float64{2.718281828459045, 7.38905609893065, 20.085536923187668, 54.598150033144236, 148.4131591025766, 403.4287934927351}, 2, 3)),
		},
		{
			name:         "add",
			fn:           Add,
			in:           ts(arng(t, 1, 7, 2, 3), arng(t, 7, 13, 2, 3)),
			expected:     ts(nd(t, []float64{8, 10, 12, 14, 16, 18}, 2, 3)),
			expectedGrad: ts(ones(t, 2, 3), ones(t, 2, 3)),
		},
		{
			name:         "add2",
			fn:           Add,
			in:           ts(arng(t, 1, 7, 2, 3), scalar(t, 2)),
			expected:     ts(nd(t, []float64{3, 4, 5, 6, 7, 8}, 2, 3)),
			expectedGrad: ts(ones(t, 2, 3), scalar(t, 6)),
		},
		{
			name:         "sub",
			fn:           Sub,
			in:           ts(arng(t, 1, 7, 2, 3), arng(t, 7, 13, 2, 3)),
			expected:     ts(nd(t, []float64{-6, -6, -6, -6, -6, -6}, 2, 3)),
			expectedGrad: ts(ones(t, 2, 3), nd(t, []float64{-1, -1, -1, -1, -1, -1}, 2, 3)),
		},
		{
			name:         "sub2",
			fn:           Sub,
			in:           ts(arng(t, 1, 7, 2, 3), scalar(t, 2)),
			expected:     ts(nd(t, []float64{-1, 0, 1, 2, 3, 4}, 2, 3)),
			expectedGrad: ts(ones(t, 2, 3), scalar(t, -6)),
		},
		{
			name:         "mul",
			fn:           Mul,
			in:           ts(arng(t, 1, 7, 2, 3), arng(t, 7, 13, 2, 3)),
			expected:     ts(nd(t, []float64{7, 16, 27, 40, 55, 72}, 2, 3)),
			expectedGrad: ts(arng(t, 7, 13, 2, 3), arng(t, 1, 7, 2, 3)),
		},
		{
			name:         "mul2",
			fn:           Mul,
			in:           ts(arng(t, 1, 7, 2, 3), scalar(t, 2)),
			expected:     ts(nd(t, []float64{2, 4, 6, 8, 10, 12}, 2, 3)),
			expectedGrad: ts(nd(t, []float64{2, 2, 2, 2, 2, 2}, 2, 3), scalar(t, 21)),
		},
		{
			name: "div",
			fn:   Div,
			in: ts(
				nd(t, []float64{2, 4, 6, 8, 10, 12}, 2, 3),
				nd(t, []float64{1, 2, 3, 4, 5, 6}, 2, 3),
			),
			expected: ts(
				nd(t, []float64{2, 2, 2, 2, 2, 2}, 2, 3)),
			expectedGrad: ts(
				nd(t, []float64{1, 0.5, 0.3333333333333333, 0.25, 0.2, 0.166666666666666666}, 2, 3),
				nd(t, []float64{-2.0, -1.0, -0.6666666666666666, -0.5, -0.4, -0.3333333333333333}, 2, 3),
			),
		},
		{
			name:         "div2",
			fn:           Div,
			in:           ts(arng(t, 1, 7, 2, 3), scalar(t, 2)),
			expected:     ts(nd(t, []float64{0.5, 1, 1.5, 2, 2.5, 3}, 2, 3)),
			expectedGrad: ts(nd(t, []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, 2, 3), scalar(t, -5.25)),
		},
		{
			name:         "neg",
			fn:           Neg,
			in:           ts(arng(t, 1, 7, 2, 3)),
			expected:     ts(nd(t, []float64{-1, -2, -3, -4, -5, -6}, 2, 3)),
			expectedGrad: ts(nd(t, []float64{-1, -1, -1, -1, -1, -1}, 2, 3)),
		},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			xs := make([]*Variable, len(tc.in))
			args := []reflect.Value{}
			for i, in := range tc.in {
				v := NewVar(in)
				xs[i] = v
				args = append(args, reflect.ValueOf(v))
			}

			for _, e := range tc.extra {
				args = append(args, reflect.ValueOf(e))
			}

			f := reflect.ValueOf(tc.fn)

			var results []reflect.Value
			if tc.variable {
				results = f.CallSlice(args)
			} else {
				results = f.Call(args)
			}

			if len(results) != 2 {
				t.Errorf("unexpected return length: %v", len(results))
			}

			// check err
			if !results[1].IsNil() {
				err, ok := results[1].Interface().(error)
				if !ok {
					t.Errorf("unexpected second-returned value: %v", results[1])
				}

				if err != nil {
					t.Errorf("unexpected err on forward: %v", err)
				}
			}

			// check returned
			y, ok := results[0].Interface().(*Variable)
			if !ok {
				t.Errorf("unexpected first-returned value: %v", results[0])
			}

			verify(t, xs, vs(y), tc.expected, tc.expectedGrad)

			// if tc.dezeroX != nil && tc.dezeroY != nil {
			// 	verifyDezeroS(t, tc.name, tc.dezeroX, tc.dezeroY, tc.expected, tc.expectedGrad)
			// }
		})
	}
}

// do this:
//   - check output is the same as expected
//   - call Backward() for each output
//   - check gradient of input is the same as expected
func verify(t *testing.T, in, out []*Variable, expected, expectedGrad []*tensor.Tensor) {
	t.Helper()

	// check calc output
	if len(expected) != len(out) {
		t.Errorf("result length mismatch expected %v but got %v", len(expected), len(out))
	}

	for i := range expected {
		if !expected[i].Equals(out[i].data) {
			t.Errorf("expected %v but got %v", expected[i], out[i].data)
		}
	}

	// check gradient
	for _, o := range out {
		if err := o.Backward(); err != nil {
			t.Errorf("unexpected err on backward: %v", err)
		}
	}

	if len(expectedGrad) != len(in) {
		t.Errorf("grad length mismatch expected %v but got %v", len(expectedGrad), len(in))
	}

	for i := range expectedGrad {
		if !expectedGrad[i].Equals(in[i].grad.data) {
			t.Errorf("grad: expected %v but got %v", expectedGrad[i].RawString(), in[i].grad.data.RawString())
		}
	}
}

// type dzr struct {
// 	y    []float64
// 	ySp  []float64
// 	ySt  []float64
// 	xG   []float64
// 	xGSp []float64
// 	xGSt []float64
// }

// 1 input, 1 output
// func verifyDezeroS(t *testing.T, name string, x, y []string, expected, expectedGrad []*tensor.Tensor) {
// 	src := `
// import dezero
// import dezero.functions as F
// import numpy as np
//
// x = dezero.Variable(%s)
// y = %s
// y.backward()
// d = y.data
// g = x.grad.data
// # use flatten to parse easily on Go side.
// print(f"{tuple(d.flatten())}_{d.shape}_{d.strides}_{tuple(g.flatten())}_{g.shape}_{g.strides}")
// `
// 	pyf := fmt.Sprintf("./python/%s.py", name)
// 	err := os.WriteFile(pyf, []byte(fmt.Sprintf(src, x, y)), 0755)
// 	if err != nil {
// 		t.Errorf("unexpected err on writing python file %s.py: %v", name, err)
// 		return
// 	}
//
// 	t.Cleanup(func() { os.Remove(pyf) })
//
// 	out, err := exec.Command("python", pyf).CombinedOutput()
// 	if err != nil {
// 		t.Errorf("unexpected err on executing python %s.py: [%v] %s", name, err, string(out))
// 		return
// 	}
//
// 	vars := strings.Split(string(out), "_")
// 	if len(vars) != 6 {
// 		t.Errorf("unexpected python output %s.py: [%v] %s", name, err, string(out))
// 		return
// 	}
//
// 	parse := func(t *testing.T, s string, strides bool) []float64 {
// 		s = strings.TrimLeft(s, "(")
// 		s = strings.TrimRight(s, ")\n")
// 		ns := strings.Split(s, ", ")
// 		result := []float64{}
// 		for _, n := range ns {
// 			f, err := strconv.ParseFloat(n, 64)
// 			if err != nil {
// 				t.Fatalf("unexpected python output %s.py: fail to parse as float %s", name, n)
// 			}
// 			if strides {
// 				// in numpy, stride is presented as bit count so convert to byte
// 				f /= 8
// 			}
// 			result = append(result, f)
// 		}
// 		return result
// 	}
//
// 	r := dzr{
// 		y:    parse(t, vars[0], false),
// 		ySp:  parse(t, vars[1], false),
// 		ySt:  parse(t, vars[2], true),
// 		xG:   parse(t, vars[3], false),
// 		xGSp: parse(t, vars[4], false),
// 		xGSt: parse(t, vars[5], true),
// 	}
//
// 	toint := func(fs []float64) []int {
// 		is := make([]int, len(fs))
// 		for i := range fs {
// 			is[i] = int(fs[i])
// 		}
// 		return is
// 	}
//
// 	check(t, expected.Data, r.y, "y")
// 	check(t, expected.CopyShape(), toint(r.ySp), "y.shape")
// 	check(t, expected.CopyStrides(), toint(r.ySt), "y.strides")
// 	check(t, expectedGrad.Data, r.xG, "x.grad")
// 	check(t, expectedGrad.CopyShape(), toint(r.xGSp), "y.grad.shape")
// 	check(t, expectedGrad.CopyStrides(), toint(r.xGSt), "y.grad.strides")
// }

func check[S ~[]E, E comparable](t *testing.T, expected, got S, name string) {
	if !slices.Equal(expected, got) {
		t.Fatalf("mismatch with python output (%s): expected: %v, got: %v", name, expected, got)
	}
}

/*
 * tensor factory helpers
 */
func scalar(t *testing.T, data float64) *tensor.Tensor {
	t.Helper()
	return tensor.FromScalar(data)
}

func nd(t *testing.T, data []float64, shape ...int) *tensor.Tensor {
	t.Helper()
	tsr, _ := tensor.Nd(data, shape...)
	return tsr
}

func arng(t *testing.T, from, to int, shape ...int) *tensor.Tensor {
	t.Helper()
	tsr, _ := tensor.ArangeFrom(from, to).Reshape(shape...)
	return tsr
}

func ones(t *testing.T, shape ...int) *tensor.Tensor {
	t.Helper()
	return tensor.Ones(shape...)
}

func trans(t *testing.T, orig *tensor.Tensor, axes ...int) *tensor.Tensor {
	t.Helper()
	tr, _ := orig.Transpose(axes...)
	return tr
}

// makes slice of tensors
func ts(tensors ...*tensor.Tensor) []*tensor.Tensor {
	return tensors
}

// makes slice of variable
func vs(vs ...*Variable) []*Variable {
	return vs
}

package tensor2

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"
	"testing/quick"
)

var _ quick.Generator = &randomIndexArg{}

type randomIndexArg struct {
	inArr   []float64
	inShape []int
	arg     []*IndexArg
}

func (_ *randomIndexArg) Generate(rand *rand.Rand, size int) reflect.Value {
	// returns rand value from 1 to n.
	non0rand := func(n int) int {
		result := 0
		for result == 0 {
			result = rand.Intn(n)
		}
		return result
	}

	// First, determine number of dimension.
	ndim := non0rand(7)

	// Second, determine the shape.
	shape := make([]int, ndim)
	for i := range ndim {
		shape[i] = non0rand(9)
	}

	// Third, create random tensor by detemined shape.
	t := Rand(shape...)
	for i, d := range t.data {
		// Round the number to 3 decimal places
		t.data[i] = math.Floor(d*1000) / 1000
	}

	// At last, determine the index randomly.
	arg := &randomIndexArg{inArr: t.data, inShape: t.Shape, arg: []*IndexArg{At(0)}}
	return reflect.ValueOf(arg)
}

func (a *randomIndexArg) String() string {
	args := ""
	for i, aa := range a.arg {
		args += aa.String()
		if i != len(a.arg)-1 {
			args += ", "
		}
	}
	return fmt.Sprintf("data: %v, shape: %v, args: [%v]", a.inArr, a.inShape, args)
}

type Result struct {
	Data  []float64
	Shape []int
}

func (r *Result) String() string {
	return fmt.Sprintf("%v %v", r.Data, r.Shape)
}

func TestIndex_quick(t *testing.T) {
	tempdir := t.TempDir()

	onTensor := func(arg *randomIndexArg) *Result {
		ten, err := NdShape(arg.inArr, arg.inShape...)
		if err != nil {
			t.Fatalf("initialize tensor: %v", err)
		}

		ten2, err := ten.Index(arg.arg...)
		if err != nil {
			t.Fatalf("index tensor: %v", err)
		}

		return &Result{Data: ten2.Flatten(), Shape: ten2.Shape}
	}

	onNumpy := func(arg *randomIndexArg) *Result {
		arr := ""
		for _, f := range arg.inArr {
			arr += fmt.Sprintf("%v, ", f)
		}
		shp := ""
		for _, i := range arg.inShape {
			shp += fmt.Sprintf("%v, ", i)
		}

		indices := ""
		for _, arg := range arg.arg {
			indices += fmt.Sprintf("%v, ", arg.numpyIndexString())
		}

		pyprg := []string{
			fmt.Sprintf("x = np.array([%s]).reshape(%s)", arr, shp),
			fmt.Sprintf("y = x[%s]", indices),
			fmt.Sprintf("print(y.flatten(), y.shape)"),
		}
		data, shape := runAsNumpyDataAndShape(t, pyprg, tempdir)
		return &Result{Data: data, Shape: shape}
	}

	err := quick.CheckEqual(onTensor, onNumpy, nil)
	if err != nil {
		cee := err.(*quick.CheckEqualError)
		t.Fatalf("quick check (#%v):\n  input       : %v\n  go output   : %v\n  numpy output: %v\n", cee.Count, cee.In, cee.Out1, cee.Out2)
	}
}

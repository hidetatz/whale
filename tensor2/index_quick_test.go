package tensor2

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
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
	ndim := non0rand(4)

	// Second, determine the shape.
	shape := make([]int, ndim)
	for i := range ndim {
		shape[i] = non0rand(3)
	}

	// Third, create random tensor by detemined shape.
	t := Rand(shape...)
	for i, d := range t.data {
		// Round the number to 3 decimal places
		t.data[i] = math.Floor(d*1000) / 1000
	}

	// At last, determine the index randomly.
	argslen := non0rand(ndim + 1)
	args := make([]*IndexArg, argslen)
	for i := range len(args) {
		dim := shape[i]
		switch rand.Intn(3) { // 0, 1, 2
		case 0:
			// type: int
			args[i] = At(rand.Intn(dim))
		case 1:
			// type: slice
			switch rand.Intn(7) {
			case 0:
				// only start
				args[i] = From(rand.Intn(dim))
			case 1:
				// only end
				args[i] = To(rand.Intn(dim))
			case 2:
				// only step
				args[i] = By(non0rand(dim + 1))
			case 3:
				// start, end
				from := rand.Intn(dim)
				to := rand.Intn(dim - from)
				args[i] = FromTo(from, to)
			case 4:
				// start, step
				from := rand.Intn(dim)
				by := non0rand(dim + 1)
				args[i] = FromBy(from, by)
			case 5:
				// end, step
				to := rand.Intn(dim)
				by := non0rand(dim + 1)
				args[i] = ToBy(to, by)
			case 6:
				// all
				from := rand.Intn(dim)
				to := rand.Intn(dim - from)
				by := non0rand(dim + 1)
				args[i] = FromToBy(from, to, by)
			}
		case 2:
			// type: list
			shp := []int{}
			switch rand.Intn(2) {
			case 0:
				shp = []int{3}
			case 1:
				shp = []int{2, 3}
			case 2:
				shp = []int{3, 2, 3}
			case 3:
				shp = []int{4, 3, 2, 3}
			}
			size := product(shp)
			data := make([]float64, size)
			for i := range len(data) {
				data[i] = float64(rand.Intn(dim))
			}
			args[i] = List(Must(NdShape(data, shp...)))
		}
	}

	arg := &randomIndexArg{inArr: t.data, inShape: t.Shape, arg: args}
	return reflect.ValueOf(arg)
}

func (a *randomIndexArg) String() string {
	in := []string{}
	for _, i := range a.inArr {
		in = append(in, fmt.Sprintf("%v", i))
	}

	shp := []string{}
	for _, s := range a.inShape {
		shp = append(shp, fmt.Sprintf("%v", s))
	}

	args := []string{}
	for _, aa := range a.arg {
		args = append(args, aa.String())
	}

	return fmt.Sprintf("data: [%v], shape: [%v], args: [%v]", strings.Join(in, ", "), strings.Join(shp, ", "), strings.Join(args, ", "))
}

type Result struct {
	Data  []float64
	Shape []int
}

func (r *Result) String() string {
	return fmt.Sprintf("%v %v", r.Data, r.Shape)
}

func TestIndex_quick(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}

	tempdir := t.TempDir()

	onTensor := func(arg *randomIndexArg) *Result {
		fmt.Println(arg)
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
		arr := []string{}
		for _, f := range arg.inArr {
			arr = append(arr, fmt.Sprintf("%v", f))
		}

		shp := []string{}
		for _, i := range arg.inShape {
			shp = append(shp, fmt.Sprintf("%v", i))
		}

		indices := []string{}
		for _, arg := range arg.arg {
			indices = append(indices, arg.numpyIndexString())
		}

		pyprg := []string{
			fmt.Sprintf("x = np.array([%s]).reshape(%s)", strings.Join(arr, ", "), strings.Join(shp, ", ")),
			fmt.Sprintf("y = x[%s]", strings.Join(indices, ", ")),
			fmt.Sprintf("print(y.flatten(), y.shape)"),
		}
		data, shape := runAsNumpyDataAndShape(t, pyprg, tempdir)
		return &Result{Data: data, Shape: shape}
	}

	err := quick.CheckEqual(onTensor, onNumpy, &quick.Config{MaxCount: 500})
	if err != nil {
		cee := err.(*quick.CheckEqualError)
		t.Fatalf("quick check (#%v):\n  input       : %v\n  go output   : %v\n  numpy output: %v\n", cee.Count, cee.In, cee.Out1, cee.Out2)
	}
}

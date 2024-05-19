package tensor

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
	inArr   []float32
	inShape []int
	arg     []*IndexArg
}

func (_ *randomIndexArg) Generate(rand *rand.Rand, size int) reflect.Value {
	return reflect.ValueOf(generateRandomIndexArg(rand, size))
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

type randomIndexUpdateArg struct {
	r      *randomIndexArg
	target *Tensor
}

func (_ *randomIndexUpdateArg) Generate(rand *rand.Rand, size int) reflect.Value {
	r := generateRandomIndexArg(rand, size)
	// todo: use more randomly generated target tensor which can be broadcasted
	target := Scalar(3)
	return reflect.ValueOf(&randomIndexUpdateArg{r: r, target: target})
}

func (a *randomIndexUpdateArg) String() string {
	return fmt.Sprintf("%v, target: [%v]", a.r.String(), a.target.OnelineString())
}

func generateRandomIndexArg(rand *rand.Rand, size int) *randomIndexArg {
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
		shape[i] = non0rand(6)
	}

	// Third, create random tensor by detemined shape.
	t := Arange(0, float32(product(shape)), 1).Reshape(shape...)
	for i, d := range t.data {
		// Round the number to 3 decimal places
		t.data[i] = float32(math.Floor(float64(d*1000))) / 1000
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
			switch rand.Intn(4) {
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
			data := make([]float32, size)
			for i := range len(data) {
				data[i] = float32(rand.Intn(dim))
			}
			args[i] = List(NdShape(data, shp...))
		}
	}

	return &randomIndexArg{inArr: t.data, inShape: t.Shape, arg: args}
}

type Result struct {
	Data  []float32
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
		ten, err := RespErr.NdShape(arg.inArr, arg.inShape...)
		if err != nil {
			t.Fatalf("initialize tensor: %v", err)
		}

		ten2 := ten.Index(arg.arg...)

		// need to resolve:
		// not sure why but sometimes ten2.Shape is returned as nil,
		// but it should be actually []int{}.
		// Because of this, comparing with numpy output fails so this check is added.
		if ten2.Shape == nil {
			ten2.Shape = []int{}
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

func TestIndexUpdate_quick(t *testing.T) {
	if testing.Short() {
		t.Skip()
	}

	tempdir := t.TempDir()

	onTensor := func(arg *randomIndexUpdateArg) *Result {
		defer func() {
			if r := recover(); r != nil {
				t.Fatalf("recovered from panic: %v", r)
			}
		}()
		ten, err := RespErr.NdShape(copySlice(arg.r.inArr), copySlice(arg.r.inShape)...)
		if err != nil {
			t.Fatalf("initialize tensor: %v", err)
		}

		ten.IndexSub(arg.r.arg, arg.target)
		return &Result{Data: ten.Flatten(), Shape: ten.Shape}
	}

	onNumpy := func(arg *randomIndexUpdateArg) *Result {
		arr := []string{}
		for _, f := range arg.r.inArr {
			arr = append(arr, fmt.Sprintf("%v", f))
		}

		shp := []string{}
		for _, i := range arg.r.inShape {
			shp = append(shp, fmt.Sprintf("%v", i))
		}

		indices := []string{}
		for _, arg := range arg.r.arg {
			indices = append(indices, arg.numpyIndexString())
		}

		pyprg := []string{
			fmt.Sprintf("x = np.array([%s]).reshape(%s)", strings.Join(arr, ", "), strings.Join(shp, ", ")),
			fmt.Sprintf("x[%s] -= 3", strings.Join(indices, ", ")),
			fmt.Sprintf("print(x.flatten(), x.shape)"),
		}
		data, shape := runAsNumpyDataAndShape(t, pyprg, tempdir)
		return &Result{Data: data, Shape: shape}
	}

	err := quick.CheckEqual(onTensor, onNumpy, &quick.Config{MaxCount: 500})
	if err != nil {
		cee, ok := err.(*quick.CheckEqualError)
		if ok {
			t.Fatalf("quick check (#%v):\n  input       : %v\n  go output   : %v\n  numpy output: %v\n", cee.Count, cee.In, cee.Out1, cee.Out2)
		} else {
			t.Fatalf(err.Error())
		}
	}
}

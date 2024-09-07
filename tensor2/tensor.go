package main

import (
	"fmt"
	"os"
	"os/exec"
	"plugin"
	"strings"
)

type op int

type _ops struct {
	constant op
	add      op
}

// Pseudo namespacing.
// Field name is not written to notice the field count mismatch on compilation.
var ops = &_ops{
	1, 2,
}

type Tensor struct {
	op   op
	src  []*Tensor
	data []float32
}

func Empty(op op) *Tensor {
	return &Tensor{op: op}
}

func New(data []float32) *Tensor {
	return &Tensor{op: ops.constant, data: data}
}

type task struct {
	op     op
	data   []float32
	inputs []int
}

func (t *Tensor) toposort() []*task {
	// todo: do toposort
	task0 := &task{op: t.src[0].src[0].op, data: t.src[0].src[0].data}
	task1 := &task{op: t.src[0].src[1].op, data: t.src[0].src[1].data}
	task2 := &task{op: t.src[0].op, inputs: []int{0, 1}}
	task3 := &task{op: t.src[1].op, data: t.src[1].data}
	task4 := &task{op: t.op, inputs: []int{2, 3}}
	return []*task{task0, task1, task2, task3, task4}
}

func (t *Tensor) Materialize() []float32 {
	params := []int{}
	prgs := []string{}
	tasks := t.toposort()
	for i, task := range tasks {
		switch task.op {
		case ops.constant:
			params = append(params, i)
		case ops.add:
			prgs = append(prgs, fmt.Sprintf("	Data%v := add(Data%v, Data%v)", i, task.inputs[0], task.inputs[1]))
		}

		if i == len(tasks)-1 {
			prgs = append(prgs, fmt.Sprintf("	Result = Data%v", i))
		}
	}

	prg := `package main

%v

var Result []float32

func add(a []float32, b []float32) []float32 {
	data := make([]float32, len(a))
	for i := range a {
		data[i] = a[i] + b[i]
	}
	return data
}

func F() {
%v
}
`

	sparams := make([]string, len(params))
	for i := range params {
		sparams[i] = fmt.Sprintf("var Data%v []float32", params[i])
	}

	f, err := os.Create("/tmp/f.go")
	if err != nil {
		panic(err)
	}

	fmt.Println(fmt.Sprintf(prg, strings.Join(sparams, "\n"), strings.Join(prgs, "\n")))

	f.WriteString(fmt.Sprintf(prg, strings.Join(sparams, "\n"), strings.Join(prgs, "\n")))
	f.Close()

	out, err := exec.Command("go", "build", "-o", "/tmp/f.so", "-buildmode=plugin", "/tmp/f.go").CombinedOutput()
	if err != nil {
		panic(string(out))
	}

	p, err := plugin.Open("/tmp/f.so")
	if err != nil {
		panic(err)
	}

	for _, param := range params {
		d, err := p.Lookup(fmt.Sprintf("Data%v", param))
		if err != nil {
			panic(err)
		}

		*d.(*[]float32) = tasks[param].data
	}

	fn, err := p.Lookup("F")
	if err != nil {
		panic(err)
	}

	fn.(func())()

	result, err := p.Lookup("Result")
	if err != nil {
		panic(err)
	}

	return *result.(*[]float32)
}

func (t *Tensor) Add(t2 *Tensor) *Tensor {
	y := Empty(ops.add)
	y.src = []*Tensor{t, t2}
	return y
}

func main() {
	t := New([]float32{1, 2})
	t2 := New([]float32{3, 4})
	t3 := t.Add(t2)
	t4 := t3.Add(New([]float32{10, 10}))

	fmt.Println(t4.Materialize())
}

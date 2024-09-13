package main

import (
	"fmt"
	"os"
	"os/exec"
	"plugin"
	"strings"
)

type golang struct{}

func (_ *golang) run(tasks []*task) []float32 {
	prgs := []string{}
	params := []int{}
	for i, task := range tasks {
		switch task.op {
		case ops.constant:
			params = append(params, i)
		case ops.add:
			prgs = append(prgs, fmt.Sprintf("	Data%v := add(Data%v, Data%v)", i, task.inputs[0], task.inputs[1]))

		case ops.mul:
			prgs = append(prgs, fmt.Sprintf("	Data%v := mul(Data%v, Data%v)", i, task.inputs[0], task.inputs[1]))
		}

		if i == len(tasks)-1 {
			prgs = append(prgs, fmt.Sprintf("	return Data%v", i))
		}
	}

	prg := `package main

%v

func add(a []float32, b []float32) []float32 {
	data := make([]float32, len(a))
	for i := range a {
		data[i] = a[i] + b[i]
	}
	return data
}

func mul(a []float32, b []float32) []float32 {
	data := make([]float32, len(a))
	for i := range a {
		data[i] = a[i] * b[i]
	}
	return data
}

func F() []float32 {
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

	program := fmt.Sprintf(prg, strings.Join(sparams, "\n"), strings.Join(prgs, "\n"))
	f.WriteString(program)
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

		*d.(*[]float32) = tasks[param].constant
	}

	fn, err := p.Lookup("F")
	if err != nil {
		panic(err)
	}

	result := fn.(func() []float32)()

	return result
}

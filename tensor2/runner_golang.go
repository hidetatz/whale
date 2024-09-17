package main

import (
	"fmt"
	"os"
	"os/exec"
	"plugin"
	"strings"
)

type golang struct {
	indentlevel int
}

func (g *golang) indent() string {
	return strings.Repeat("	", g.indentlevel)
}

func (g *golang) f(format string, a ...any) string {
	return g.indent() + fmt.Sprintf(format, a...)
}

func (g *golang) dataOnHost(idx int) []string {
	return []string{
		g.f("var Data%v []float32", idx),
	}
}

func (g *golang) alu2(fn string, length, result, left, right int) []string {
	return []string{
		g.f("var Data%v []float32", result),
		g.f("Data%v = make([]float32, %v)", result, length),
		g.f("%v(Data%v, Data%v, Data%v);", fn, left, right, result),
	}
}

func (g *golang) returnresult(idx int) []string {
	return []string{
		g.f("return Data%v", idx),
	}
}

func (g *golang) run(tasks []*task) []float32 {
	resultlen := len(tasks[0].constant) // todo: calculate properly

	computes := [][]string{}
	inputCPU := [][]string{}
	inputIdx := []int{}

	var returnresult []string

	for i, task := range tasks {
		switch task.op {
		case ops.constant:
			inputIdx = append(inputIdx, i)
			inputCPU = append(inputCPU, g.dataOnHost(i))

		case ops.add:
			computes = append(computes, g.alu2("add", resultlen, i, task.inputs[0], task.inputs[1]))

		case ops.mul:
			computes = append(computes, g.alu2("mul", resultlen, i, task.inputs[0], task.inputs[1]))
		}

		if i == len(tasks)-1 {
			returnresult = g.returnresult(i)
		}
	}

	prg := `package main

%v

func add(a, b, c []float32) {
	for i := range a {
		c[i] = a[i] + b[i]
	}
}

func mul(a, b, c []float32)  {
	for i := range a {
		c[i] = a[i] * b[i]
	}
}

func F() []float32 {
	// computes
	%s

	// return result
	%s
}
`

	f, err := os.Create("/tmp/f.go")
	if err != nil {
		panic(err)
	}

	join1 := func(ss []string, sep string) string {
		return strings.Join(ss, sep)
	}

	join2 := func(sss [][]string, sep2, sep1 string) string {
		s := make([]string, len(sss))
		for i := range sss {
			s[i] = join1(sss[i], sep1)
		}
		return strings.Join(s, sep2)
	}

	program := fmt.Sprintf(prg,
		join2(inputCPU, "\n", "\n"),
		join2(computes, "\n\n	", "\n	"),
		join1(returnresult, "\n	"),
	)

	f.WriteString(program)
	f.Close()

	out, err := exec.Command("go", "build", "-o", "/tmp/whale_f_golang.so", "-buildmode=plugin", "/tmp/f.go").CombinedOutput()
	if err != nil {
		panic(string(out))
	}

	p, err := plugin.Open("/tmp/whale_f_golang.so")
	if err != nil {
		panic(err)
	}

	for _, param := range inputIdx {
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

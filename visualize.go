package whale

import (
	"fmt"
	"os"
	"os/exec"
	"unsafe"
)

// VisualizeGraph outputs the calculation graph as graph.png on the current directory.
func VisualizeGraph(v *Variable) error {
	pv := func(v *Variable) string { return fmt.Sprintf("%d", uintptr(unsafe.Pointer(v))) }
	pf := func(f *function) string { return fmt.Sprintf("%d", uintptr(unsafe.Pointer(f))) }

	varToDot := func(v *Variable) string {
		return fmt.Sprintf("%s [label=\"data %.2f | grad %.2f\", shape=box]\n", pv(v), v.data, v.grad.data)
	}

	funcToDot := func(f *function) string {
		txt := fmt.Sprintf("%s [label=\"%s\"]\n", pf(f), f.String())
		for _, x := range f.inputs {
			txt += fmt.Sprintf("%s -> %s\n", pv(x), pf(f))
		}
		for _, y := range f.outputs {
			txt += fmt.Sprintf("%s -> %s\n", pf(f), pv(y))
		}
		return txt
	}

	txt := ""
	fs := []*function{}
	uniqueadd := func(f *function) {
		for _, added := range fs {
			if added == f {
				return
			}
		}
		fs = append(fs, f)
	}

	uniqueadd(v.creator)
	txt += varToDot(v)
	for len(fs) > 0 {
		var f *function
		f, fs = fs[len(fs)-1], fs[:len(fs)-1] // pop last

		txt += funcToDot(f)
		for _, x := range f.inputs {
			txt += varToDot(x)
			if x.creator != nil {
				uniqueadd(x.creator)
			}
		}
	}

	graphfmt := `
digraph g {
  graph [
    charset = "UTF-8";
    rankdir = LR,
  ];

  %s
} `
	dotSrc := fmt.Sprintf(graphfmt, txt)

	os.WriteFile("./graph.dot", []byte(dotSrc), 0755)
	out, err := exec.Command("dot", "./graph.dot", "-T", "png", "-o", "graph.png").Output()
	if err != nil {
		return fmt.Errorf("%s: %s", err.Error(), string(out))
	}
	os.Remove("./graph.dot")

	return nil
}

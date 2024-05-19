package whale

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"
)

type Plot struct {
	base  []string
	cmds  []string
	plots []string
	data  []struct {
		x []float32
		y []float32
	}
}

func NewPlot() *Plot {
	return &Plot{
		cmds: []string{"set key noautotitle"},
	}
}

func (p *Plot) addData(x, y []float32) {
	var xs, ys []float32
	for i := range x {
		xs = append(xs, x[i])
		ys = append(ys, y[i])
	}

	p.data = append(p.data, struct {
		x []float32
		y []float32
	}{
		x: xs,
		y: ys,
	})
}

func (p *Plot) Scatter(x, y []float32, color string) error {
	if len(x) != len(y) {
		return fmt.Errorf("scatter: x and y must be the same size")
	}

	p.plots = append(p.plots, fmt.Sprintf("'-' with points linecolor rgb '%s'", color))
	p.addData(x, y)
	return nil
}

func (p *Plot) Line(x, y []float32, color string) error {
	if len(x) != len(y) {
		return fmt.Errorf("line: x and y must be the same size")
	}

	p.plots = append(p.plots, fmt.Sprintf("'-' with lines linecolor rgb '%s'", color))
	p.addData(x, y)
	return nil
}

func (p *Plot) Exec() error {
	cmd := exec.Command("gnuplot", "-persist")

	gnuplot := ""
	gnuplot += strings.Join(p.cmds, "; ")
	gnuplot += "; plot " + strings.Join(p.plots, ", ")
	gnuplot += "\n"
	for _, d := range p.data {
		for i := range d.x {
			gnuplot += fmt.Sprintf("%.3f, %.3f\n", d.x[i], d.y[i])
		}
		gnuplot += "e\n"
	}

	var in bytes.Buffer
	in.WriteString(gnuplot)

	cmd.Stdin = &in

	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("run gnuplot: %s", string(out))
	}

	return nil
}

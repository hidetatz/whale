package tensor2

import (
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// the program must output target ndarray data and shape.
// e.g. "print(y.flatten(), y.shape)"
func runAsNumpyDataAndShape(t *testing.T, prog []string) ([]float64, []int) {
	t.Helper()
	out := execNumpy(t, prog)
	return parseNumpyDataAndShape(t, out)
}

func execNumpy(t *testing.T, prog []string) string {
	t.Helper()

	prog = slices.Concat([]string{"import numpy as np"}, prog)
	progstr := strings.Join(prog, "; ")

	cmd := exec.Command("python", "-c", progstr)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("running python err: %v: %v", err, out)
	}

	return strings.TrimSpace(string(out))
}

// The output should look like:
// [0 1 2 3 4 5] (2, 3)
func parseNumpyDataAndShape(t *testing.T, output string) ([]float64, []int) {
	t.Helper()

	split := strings.Split(output, "] (")
	if len(split) != 2 {
		t.Fatalf("unexpected output format: %v", output)
	}

	npdata, npshape := split[0], split[1]

	npdata = strings.TrimSpace(strings.TrimLeft(npdata, "["))
	data := []float64{}
	for _, d := range strings.Split(npdata, " ") {
		trim := strings.TrimSpace(d)
		if trim == "" {
			continue
		}

		f, err := strconv.ParseFloat(trim, 64)
		if err != nil {
			t.Fatalf("parse data as float64: err: '%v', d: '%v', data: '%v', output: '%v'", err, d, npdata, output)
		}
		data = append(data, f)
	}

	npshape = strings.TrimSpace(strings.TrimRight(npshape, "),"))
	shape := []int{}

	// if scalar, shape will be empty
	if npshape == "" {
		return data, shape
	}

	for _, s := range strings.Split(npshape, ",") {
		i, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
		if err != nil {
			t.Fatalf("parse shape as int64: err: '%v', s: '%v', shape: '%v', output: '%v'", err, s, npshape, output)
		}
		shape = append(shape, int(i))
	}

	return data, shape
}

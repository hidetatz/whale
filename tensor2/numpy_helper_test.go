package tensor2

import (
	"os"
	"os/exec"
	"slices"
	"strconv"
	"strings"
	"testing"
)

// the program must output target ndarray data and shape.
// e.g. "print(y.flatten(), y.shape)"
func runAsNumpyDataAndShape(t *testing.T, prog []string, tempdir string) ([]float64, []int) {
	t.Helper()
	out := execNumpy(t, prog, tempdir)
	return parseNumpyDataAndShape(t, out)
}

func execNumpy(t *testing.T, prog []string, tempdir string) string {
	t.Helper()

	prog = slices.Concat([]string{
		"import numpy as np",
		"import sys",
		"np.set_printoptions(threshold=sys.maxsize)",
	}, prog)

	progstr := strings.Join(prog, "\n")

	f, err := os.CreateTemp(tempdir, "")
	if err != nil {
		t.Fatalf("create temporary python file: %v", err)
	}

	if err := os.WriteFile(f.Name(), []byte(progstr), 0755); err != nil {
		t.Fatalf("write python file: %v", err)
	}

	cmd := exec.Command("python", f.Name())
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("running python err: %v: %v", err, string(out))
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

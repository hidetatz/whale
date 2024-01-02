package whale

import (
	"fmt"
	"os/exec"
)

// Scatter shows the scatter graph using gnuplot command.
func Scatter(x, y []float64) error {
	if len(x) != len(y) {
		return fmt.Errorf("scatter: x and y must be the same size")
	}

	cmd := ""
	for i := range x {
		cmd += fmt.Sprintf(`"<echo '%.3f %.3f'" with points ls 1`, x[i], y[i])
		if i < len(x)-1 {
			cmd += ", "
		}
	}
	err := exec.Command("gnuplot", "-p", "-e", fmt.Sprintf("set key noautotitle; set style line 1 lc rgb 'blue' pt 7; plot %s", cmd)).Run()
	if err != nil {
		return fmt.Errorf("scatter: %w", err)
	}
	return nil
}

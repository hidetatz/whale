package main

import (
	"os"
	"os/exec"
)

type backend interface {
	run(tasks []*task) []float32
}

var runner backend

func initRunner() {
	switch {
	case os.Getenv("WHALE_GO") == "1":
		runner = &golang{}
		return
	case os.Getenv("WHALE_CUDA") == "1":
		runner = &cuda{}
		return
	}

	available := func(cmd string) bool {
		err := exec.Command("bash", "-c", "command -v "+cmd).Run()
		return err == nil // likely the cmd is available
	}

	if available("nvcc") {
		runner = &cuda{}
		return
	}

	runner = &golang{}
}

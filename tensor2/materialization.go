package main

import (
	"slices"
)

type task struct {
	op       op
	constant []float32
	inputs   []int
}

func flatten(leaf *Tensor) []*Tensor {
	visited := make(map[*Tensor]bool)
	var graphs []*Tensor
	var dfs func(*Tensor)

	dfs = func(p *Tensor) {
		if visited[p] {
			return
		}

		visited[p] = true
		for _, s := range p.inputs {
			dfs(s)
		}
		graphs = append(graphs, p)
	}
	dfs(leaf)
	return graphs
}

func dependencies(graphs []*Tensor) [][]*Tensor {
	list := make([][]*Tensor, len(graphs))
	for i := range graphs {
		list[i] = graphs[i].inputs
	}
	return list
}

func calcIndegree(graphs []*Tensor) []int {
	list := make([]int, len(graphs))
	for _, graph := range graphs {
		for _, src := range graph.inputs {
			for i, t := range graphs {
				if t == src {
					list[i]++
					break
				}
			}
		}
	}
	return list
}

// linearize transforms tensor dependency tree from t (including t) to
// a list of task.
// Internally using topological sort.
func (t *Tensor) linearize() []*task {
	/*
	 * Topological sort preparation
	 */

	// Extract tensor tree to list. They are still connected via its pointer.
	flattened := flatten(t)

	// Tensors order matters, This is latter used as something like task's ID.
	// Preserve order here for later use.
	at := map[*Tensor]int{}
	for i, p := range flattened {
		at[p] = i
	}

	// Dependency list for topological sort.
	// This is a list whose length is the same as tensors.
	// deps[i] is a list of tensors that tensors[i] is depending on.
	deps := dependencies(flattened)

	// Indegree counts for topological sort.
	// This is a list whose length is the same as tensors.
	// indegrees[i] is how many tensors are depending on tensors[i].
	indegrees := calcIndegree(flattened)

	/*
	 * Topological sort.
	 */

	queue := []*Tensor{}
	for i, indegree := range indegrees {
		if indegree == 0 {
			queue = append(queue, flattened[i])
		}
	}

	result := []*Tensor{}

	// breadth first search
	for len(queue) != 0 {
		// pop left
		v := queue[0]
		queue = queue[1:]

		result = append(result, v)

		for _, dep := range deps[at[v]] {
			indegrees[at[dep]]--
			if indegrees[at[dep]] == 0 {
				queue = append(queue, flattened[at[dep]])
			}
		}
	}

	// After topological sort, the result order is from dependent to independent.
	// This must be reversed.
	slices.Reverse(result)

	tasks := make([]*task, len(result))
	for i, r := range result {
		tasks[i] = &task{op: r.op}

		if r.op == ops.constant {
			tasks[i].constant = r.data
			continue
		}

		inputs := make([]int, len(r.inputs))

		// find inputs index
		for i, dep := range r.inputs {
			for j, r := range result {
				if dep == r {
					inputs[i] = j
				}
			}
		}

		tasks[i].inputs = inputs
	}

	return tasks
}

func (t *Tensor) Materialize() []float32 {
	t.data = runner.run(t.linearize())
	return t.data
}

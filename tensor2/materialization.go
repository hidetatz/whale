package main

import (
	"slices"
)

type task struct {
	op       nodeop
	constant []float32
	inputs   []int
}

func flatten(leaf *Tensor) []*node {
	visited := make(map[*node]bool)
	var graphs []*node
	var dfs func(*node)

	dfs = func(p *node) {
		if visited[p] {
			return
		}

		visited[p] = true
		for _, s := range p.input {
			dfs(s)
		}
		graphs = append(graphs, p)
	}
	dfs(leaf.node)
	return graphs
}

func dependencies(graphs []*node) [][]*node {
	list := make([][]*node, len(graphs))
	for i := range graphs {
		list[i] = graphs[i].input
	}
	return list
}

func calcIndegree(graphs []*node) []int {
	list := make([]int, len(graphs))
	for _, graph := range graphs {
		for _, src := range graph.input {
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
	graphs := flatten(t)

	// Tensors order matters, This is latter used as something like task's ID.
	// Preserve order here for later use.
	at := map[*node]int{}
	for i, p := range graphs {
		at[p] = i
	}

	// Dependency list for topological sort.
	// This is a list whose length is the same as tensors.
	// deps[i] is a list of tensors that tensors[i] is depending on.
	deps := dependencies(graphs)

	// Indegree counts for topological sort.
	// This is a list whose length is the same as tensors.
	// indegrees[i] is how many tensors are depending on tensors[i].
	indegrees := calcIndegree(graphs)

	/*
	 * Topological sort.
	 */

	queue := []*node{}
	for i, indegree := range indegrees {
		if indegree == 0 {
			queue = append(queue, graphs[i])
		}
	}

	result := []*node{}

	// breadth first search
	for len(queue) != 0 {
		// pop left
		v := queue[0]
		queue = queue[1:]

		result = append(result, v)

		for _, dep := range deps[at[v]] {
			indegrees[at[dep]]--
			if indegrees[at[dep]] == 0 {
				queue = append(queue, graphs[at[dep]])
			}
		}
	}

	// After topological sort, the result order is from dependent to independent.
	// This must be reversed.
	slices.Reverse(result)

	tasks := make([]*task, len(result))
	for i, r := range result {
		tasks[i] = &task{op: r.op}

		if r.op == nodeops.constant {
			tasks[i].constant = r.constant
			continue
		}

		inputs := make([]int, len(r.input))

		// find inputs index
		for i, dep := range r.input {
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
	// convert tree into task list
	tasks := t.linearize()

	// render source code from tasks then run it
	result := runner.run(tasks)

	// set calculated value, materialization done
	t.data = result

	return t.data
}

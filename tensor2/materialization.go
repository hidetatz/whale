package main

import (
	"slices"
)

type task struct {
	op       op
	constant []float32
	inputs   []int
}

func flatten(leaf *Tensor) []*recipe {
	visited := make(map[*recipe]bool)
	var recipes []*recipe
	var dfs func(*recipe)

	dfs = func(r *recipe) {
		if visited[r] {
			return
		}

		visited[r] = true
		for _, s := range r.src {
			dfs(s)
		}
		recipes = append(recipes, r)
	}
	dfs(leaf.recipe)
	return recipes
}

func dependencies(recipes []*recipe) [][]*recipe {
	list := make([][]*recipe, len(recipes))
	for i := range recipes {
		list[i] = recipes[i].src
	}
	return list
}

func calcIndegree(recipes []*recipe) []int {
	list := make([]int, len(recipes))
	for _, recipe := range recipes {
		for _, src := range recipe.src {
			for i, t := range recipes {
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
	recipes := flatten(t)

	// Tensors order matters, This is latter used as something like task's ID.
	// Preserve order here for later use.
	at := map[*recipe]int{}
	for i, r := range recipes {
		at[r] = i
	}

	// Dependency list for topological sort.
	// This is a list whose length is the same as tensors.
	// deps[i] is a list of tensors that tensors[i] is depending on.
	deps := dependencies(recipes)

	// Indegree counts for topological sort.
	// This is a list whose length is the same as tensors.
	// indegrees[i] is how many tensors are depending on tensors[i].
	indegrees := calcIndegree(recipes)

	/*
	 * Topological sort.
	 */

	queue := []*recipe{}
	for i, indegree := range indegrees {
		if indegree == 0 {
			queue = append(queue, recipes[i])
		}
	}

	result := []*recipe{}

	// breadth first search
	for len(queue) != 0 {
		// pop left
		v := queue[0]
		queue = queue[1:]

		result = append(result, v)

		for _, dep := range deps[at[v]] {
			indegrees[at[dep]]--
			if indegrees[at[dep]] == 0 {
				queue = append(queue, recipes[at[dep]])
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
			tasks[i].constant = r.constant
			continue
		}

		inputs := make([]int, len(r.src))

		// find inputs index
		for i, dep := range r.src {
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
	/*
	 * Materialization overview:
	 *
	 * So every tensor has 0 or more source tensors (t.src).
	 * Let's assume calculation like this:
	 *
	 * t1 := Tensor([1, 2, 3])
	 * t2 := Tensor([4, 5, 6])
	 * t3 := t1 + t2
	 * t4 := Tensor([7, 8, 9])
	 * t5 := t3 + t4
	 *
	 * After this calculation, the tensor dependency tree will be constructed like this:
	 *
	 * t5 {
	 * 	 op: +,
	 * 	 src: [
	 *     t3: {
	 * 	     op: +,
	 * 	     src: [
	 * 	  	   t1: {
	 * 	  	     op: const,
	 * 	  	     data: [1, 2, 3],
	 * 	  	   },
	 * 	  	   t2: {
	 * 	  	     op: const,
	 * 	  	     data: [4, 5, 6],
	 * 	  	   }
	 * 	  	 ]
	 *     },
	 * 	   t4: {
	 * 	     op: const,
	 * 	     data: [7, 8, 9],
	 * 	   },
	 *   ]
	 * }
	 *
	 * When t5.Materialize() is called, this tree is converted into a linear task list like this:
	 * {
	 *   task[op: const, data: [1, 2, 3]],
	 *   task[op: const, data: [4, 5, 6]],
	 *   task[op:     +, inputs:  [0, 1]], // task index
	 *   task[op: const, data: [7, 8, 9]],
	 *   task[op:     +, inputs:  [2, 4]],
	 * }
	 *
	 * Note that it is ok to change tasks[2] and tasks[3] order.
	 *
	 * This tasks are rendered as source code which is run on device (with some device specific flavor) like this:
	 *
	 * // constant values are set from Go as pointer for code reusability
	 * var data0;
	 * var data1;
	 * var data3;
	 * float* f() {
	 *   data2 = data0 + data1
	 *   data4 = data0 + data1
	 *   return data4
	 * }
	 *
	 * Then, returned value will be set as t.data.
	 */

	// convert tree into task list
	tasks := t.linearize()

	// render source code from tasks then run it
	result := runner.run(tasks)

	// set calculated value, materialization done
	t.data = result

	return t.data
}

package tensor2

// returns x[0] * x[1] * x[2] * ...
func product(x []int) int {
	p := 1
	for _, dim := range x {
		p *= dim
	}
	return p
}

func copySlice(s []int) []int {
	c := make([]int, len(s))
	copy(c, s)
	return c
}

func all(v, length int) []int {
	r := make([]int, length)
	for i := range length {
		r[i] = v
	}
	return r
}

func toint(fs []float64) []int {
	r := make([]int, len(fs))
	for i := range fs {
		r[i] = int(fs[i])
	}
	return r
}

func seq[T int | float64](from, to T) []T {
	r := make([]T, int(to-from))
	for i := from; i < to; i += 1 {
		r[int(i-from)] = i
	}
	return r
}

// generates cartesian product of the given slice of slice.
// Let's say a is [[0, 1], [0, 1, 2], [0, 1]], returned will be:
// [
//
//	[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1],
//	[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1],
//
// ]
// If a is empty, empty list will be returned.
// If a contains a 0, empty list will be returned.
func cartesians(a [][]int) [][]int {
	if len(a) == 0 {
		return [][]int{}
	}

	var result [][]int
	var current []int
	var f func(int)
	f = func(pos int) {
		if pos == len(a) {
			temp := make([]int, len(current))
			copy(temp, current)
			result = append(result, temp)
			return
		}
		for _, n := range a[pos] {
			current = append(current, n)
			f(pos + 1)
			current = current[:len(current)-1]
		}
	}
	f(0)
	return result
}

func cartesiansIdx(a [][]int) [][]*IndexArg {
	c := cartesians(a)
	args := make([][]*IndexArg, len(c))
	for i := range c {
		args[i] = intsToIndices(c[i])
	}
	return args
}

// creates cartesian product, but regards given a as
// index slice. For example, if a is [2, 3, 2],
// this function actually returns cartesians([[0, 1], [0, 1, 2], [0, 1]]).
func cartesian(a []int) [][]int {
	arg := make([][]int, len(a))
	for i, n := range a {
		s := make([]int, n)
		for j := range n {
			s[j] = j
		}
		arg[i] = s
	}
	return cartesians(arg)
}

func cartesianIdx(a []int) [][]*IndexArg {
	c := cartesian(a)
	args := make([][]*IndexArg, len(c))
	for i := range c {
		args[i] = intsToIndices(c[i])
	}
	return args
}

func intsToIndices(ints []int) []*IndexArg {
	arg := make([]*IndexArg, len(ints))
	for i := range ints {
		arg[i] = At(ints[i])
	}
	return arg
}

func Must[T any](obj T, err error) T {
	if err != nil {
		panic(err)
	}

	return obj
}

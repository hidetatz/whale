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

func seqf(from, to int) []float64 {
	r := make([]float64, to-from)
	for i := from; i < to; i++ {
		r[i-from] = float64(i)
	}
	return r
}

// generates cartesian product of the given slice.
// Let's say a is [2, 3, 2], returned will be:
// [
//
//	[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 2, 0], [0, 2, 1],
//	[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1], [1, 2, 0], [1, 2, 1],
//
// ]
// If a is empty, empty list will be returned.
// If a contains a 0, empty list will be returned.
func cartesian(a []int) [][]int {
	if len(a) == 0 {
		return [][]int{}
	}

	var result [][]int
	var current []int
	var generate func(int)
	generate = func(pos int) {
		if pos == len(a) {
			temp := make([]int, len(current))
			copy(temp, current)
			result = append(result, temp)
			return
		}
		for i := 0; i < a[pos]; i++ {
			current = append(current, i)
			generate(pos + 1)
			current = current[:len(current)-1]
		}
	}
	generate(0)
	return result
}

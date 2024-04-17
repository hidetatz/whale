package main

import (
	"fmt"
	// "github.com/hidetatz/whale/tensor2"
)

func main() {
	a := [][]int{
		[]int{0, 0, 0, 0, 0, 0},
		[]int{1},
		[]int{0, 0, 0},
	}

	fmt.Println(cartesians(a))
}

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

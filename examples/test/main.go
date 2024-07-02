package main

import (
	"fmt"

	"github.com/hidetatz/whale/cpuid"
	"github.com/hidetatz/whale/flops"
)

func main() {
	info := cpuid.CPUID()
	fmt.Printf("%+v\n", info)
	fmt.Printf("%+v\n", flops.Calc(info))
}

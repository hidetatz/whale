package main

import (
	"fmt"

	"github.com/hidetatz/whale/cpuid"
	"github.com/hidetatz/whale/flops"
)

func main() {
	fmt.Printf("%+v\n", cpuid.CPUID())
	fmt.Printf("%+v\n", flops.Calc())
}

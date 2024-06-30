package main

import (
	"fmt"

	"github.com/hidetatz/whale/cpuid"
)

func main() {
	fmt.Printf("%+v\n", cpuid.CPUID())
}

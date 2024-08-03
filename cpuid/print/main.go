package main

import (
	"encoding/json"
	"fmt"

	"github.com/hidetatz/whale/cpuid"
)

func main() {
	j, _ := json.MarshalIndent(cpuid.CPUID(), "", "\t")
	fmt.Println(string(j))
}

package cpuid

import (
	"fmt"
)

func asmcpuid(op uint32) (eax, ebx, ecx, edx uint32)

func vendorid() string {
	_, ebx, ecx, edx := asmcpuid(0)
	return string([]byte{
		// vendor id is constructed as ebx -> edx -> ecx
		byte(ebx >> 0), byte(ebx >> 8), byte(ebx >> 16), byte(ebx >> 24),
		byte(edx >> 0), byte(edx >> 8), byte(edx >> 16), byte(edx >> 24),
		byte(ecx >> 0), byte(ecx >> 8), byte(ecx >> 16), byte(ecx >> 24),
	})
}

func CPUID() {
	fmt.Println(vendorid())
}

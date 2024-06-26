package cpuid

import (
	"fmt"
)

func asmcpuid(op uint32) (eax, ebx, ecx, edx uint32)

type CPUInfo struct {
	VendorID string
	ModelName string
}

// returns [hi:lo] bit
func bits(val uint32, hi, lo int) uint32 {
	return (val >> lo) & ((1 << (hi - lo + 1)) - 1)
}

func bitsB(val uint32, hi, lo int) byte {
	return byte(bits(val, hi, lo))
}

func asstr(u uint32) string {
	return string([]byte{bitsB(u, 8, 0), bitsB(u, 15, 8), bitsB(u, 23, 16), bitsB(u, 31, 24)})
}

func setvendorid(info *CPUInfo) {
	_, ebx, ecx, edx := asmcpuid(0x0)
	// vendor id is constructed as ebx -> edx -> ecx
	info.VendorID= asstr(ebx) + asstr(edx) + asstr(ecx)
}

func setmodelname(info *CPUInfo) {
	write := func(vals ...uint32) {
		for _, val := range vals {
			if val != 0x00 { // omit null
				info.ModelName += asstr(val)
			}
		}
	}

	write(asmcpuid(0x80000002))
	write(asmcpuid(0x80000003))
	write(asmcpuid(0x80000004))
}

func CPUID() {
	info := &CPUInfo{}
	setvendorid(info)
	setmodelname(info)

	fmt.Printf("%#v\n", info)
}

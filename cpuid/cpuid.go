package cpuid

import (
	"fmt"
)

func asmcpuid1(leaf uint32) (eax, ebx, ecx, edx uint32)
func asmcpuid2(leaf, subleaf uint32) (eax, ebx, ecx, edx uint32)

type CPUInfo struct {
	VendorID  string
	ModelName string

	// freq
	ProcessorBaseFreqMHz int
	MaxTurboFreqMHz      int

	// cores
	ThreadsPerCore int
	LogicalCores   int
	PhysicalCores  int

	// caches
	CacheLineBytes int
	// caches are usually constructed like this
	L1ICache int
	L1DCache int
	L2Cache  int
	L3Cache  int

	// only whale-related features exist here
	Supported struct {
		MMX     bool
		MMX_ext bool
		SSE     bool
		SSE2    bool
		SSE3    bool
		SSSE3   bool
		SSE4_1  bool
		SSE4_2  bool

		AVX            bool
		AVX_vnni       bool
		AVX2           bool
		AVX_vnni_int8  bool
		AVX_vnni_int16 bool
		AVX_ne_convert bool
		AVX_ifma       bool

		// TODO: add AVX512 and AMX when I obtain $4000 Xeon Processor
	}
}

type features struct {
}

func bit(val uint32, pos int) uint32 {
	return (val >> pos) & 1
}

func bitbool(val uint32, pos int) bool {
	return bit(val, pos) == uint32(1)
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

func printx(u uint32) {
	fmt.Printf("0x%x\n", u)
}

func printb(u uint32) {
	fmt.Printf("0b%b\n", u)
}

func CPUID() *CPUInfo {
	// Note: This implementation is not sufficient for some old processors.

	info := &CPUInfo{}

	// vendor id
	_, b, c, d := asmcpuid1(0x0)
	info.VendorID = asstr(b) + asstr(d) + asstr(c)

	// frequencies
	// note: For some CPU, EAX should be 0x15 to retrieve Crystal freq.
	// My processor is ok to use 0x16.
	a, b, _, _ := asmcpuid1(0x16)
	info.ProcessorBaseFreqMHz = int(bits(a, 15, 0))
	info.MaxTurboFreqMHz = int(bits(b, 15, 0))

	// cores
	_, b, _, _ = asmcpuid1(0xb)
	info.ThreadsPerCore = int(bits(b, 8, 0))
	_, b, _, _ = asmcpuid2(0xb, 1)
	info.LogicalCores = int(bits(b, 8, 0))
	info.PhysicalCores = info.LogicalCores / info.ThreadsPerCore

	// cache
	_, b, _, _ = asmcpuid1(0x1)
	info.CacheLineBytes = int(bits(b, 15, 8)) * 8 // CLFLUSH line size
	for i := uint32(0); ; i++ {
		a, b, c, d = asmcpuid2(0x4, i)
		cacheType := bits(a, 4, 0)
		if cacheType == 0 {
			break // no more caches
		}

		level := bits(a, 7, 5)
		size := int((bits(b, 11, 0) + 1) * (bits(b, 21, 12) + 1) * (bits(b, 31, 22) + 1) * (c + 1))

		switch level {
		case 1:
			switch cacheType {
			case 1:
				info.L1DCache = size
			case 2:
				info.L1ICache = size
			}
		case 2:
			// note: in case level is 2 or 3, the cache type was 3 (unified cache) on my machine.
			info.L2Cache = size
		case 3:
			info.L3Cache = size
		}
	}

	// model name
	writeModelName := func(vals ...uint32) {
		for _, val := range vals {
			if val != 0x00 { // omit null
				info.ModelName += asstr(val)
			}
		}
	}

	writeModelName(asmcpuid1(0x80000002))
	writeModelName(asmcpuid1(0x80000003))
	writeModelName(asmcpuid1(0x80000004))

	// features
	_, _, _, d = asmcpuid1(0x80000001)
	info.Supported.MMX = bitbool(d, 23)
	info.Supported.MMX_ext = bitbool(d, 22)

	_, _, c, d = asmcpuid1(0x1)
	info.Supported.SSE = bitbool(d, 25)
	info.Supported.SSE2 = bitbool(d, 26)
	info.Supported.SSE3 = bitbool(c, 0)
	info.Supported.SSSE3 = bitbool(c, 1)
	info.Supported.SSE4_1 = bitbool(c, 19)
	info.Supported.SSE4_2 = bitbool(c, 20)
	info.Supported.AVX = bitbool(c, 28)

	_, b, _, _ = asmcpuid2(0x7, 0x0)
	info.Supported.AVX2 = bitbool(b, 5)

	a, _, _, d = asmcpuid2(0x7, 0x1)
	info.Supported.AVX_vnni = bitbool(a, 4)
	info.Supported.AVX_vnni_int8 = bitbool(d, 4)
	info.Supported.AVX_vnni_int16 = bitbool(d, 10)
	info.Supported.AVX_ne_convert = bitbool(d, 5)
	info.Supported.AVX_ifma = bitbool(a, 23)

	return info
}

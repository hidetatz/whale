package cpuid

import (
	"fmt"
)

func asmcpuid(op uint32) (eax, ebx, ecx, edx uint32)

type CPUInfo struct {
	VendorID          string
	ModelName         string
	SupportedFeatures *features
}

// features defines supported features by a CPU, but only limited features related whale exist here.
type features struct {
	PrefetchWT1 bool
	Prefetchi   bool

	MMX    bool
	SSE    bool
	SSE2   bool
	SSE3   bool
	SSSE3  bool
	SSE4_1 bool
	SSE4_2 bool

	AVX            bool
	AVX2           bool
	AVX_vnni_int8  bool
	AVX_ne_convert bool
	AVX_vnni_int16 bool
	AVX_ifma       bool

	AVX512_f            bool
	AVX512_dq           bool
	AVX512_ifma         bool
	AVX512_pf           bool
	AVX512_er           bool
	AVX512_cd           bool
	AVX512_bw           bool
	AVX512_vl           bool
	AVX512_vbmi         bool
	AVX512_vbmi2        bool
	AVX512_vnni         bool
	AVX512_bitalg       bool
	AVX512_vpopcntdq    bool
	AVX512_4vnniw       bool
	AVX512_4fmaps       bool
	AVX512_vp2intersect bool
	AVX512_fp16         bool
	AVX512_bf16         bool

	AMX_bf16    bool
	AMX_tile    bool
	AMX_int8    bool
	AMX_fp16    bool
	AMX_complex bool

	AVX10 bool
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
	info.VendorID = asstr(ebx) + asstr(edx) + asstr(ecx)
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

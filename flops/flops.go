package flops

import "github.com/hidetatz/whale/cpuid"

type Flops struct {
	MFlopsFloatBase   float64
	MFlopsFloatTurbo  float64
	MFlopsDoubleBase  float64
	MFlopsDoubleTurbo float64
}

func Calc() *Flops {
	info := cpuid.CPUID()

	basefreq := float64(info.ProcessorBaseFreqMHz)
	turbofreq := float64(info.MaxTurboFreqMHz)

	cores := float64(info.LogicalCores)

	// TODO: this might not be correct which is assuming each logical core has a fpu.
	// How to detect number of fpu?
	fpus := float64(info.LogicalCores)

	fpuops := float64(1)
	if info.Supported.FMA {
		fpuops = 2 // if FMA is supported, it can compute both ADD and MUL in 1 op.
	}

	vectorlen := float64(1)
	switch {
	// must be ordered vectorlen desc

	// ymm registers
	case info.Supported.AVX2:
		vectorlen = 256
	case info.Supported.AVX:
		vectorlen = 256

	// xmm registers
	case info.Supported.SSE2:
		vectorlen = 128
	case info.Supported.SSE:
		vectorlen = 128

	// mm registers
	case info.Supported.MMX:
		vectorlen = 64
	}

	return &Flops{
		MFlopsFloatBase:   cores * fpus * fpuops * (vectorlen / (8 * 32)) * basefreq,
		MFlopsFloatTurbo:  cores * fpus * fpuops * (vectorlen / (8 * 32)) * turbofreq,
		MFlopsDoubleBase:  cores * fpus * fpuops * (vectorlen / (8 * 64)) * basefreq,
		MFlopsDoubleTurbo: cores * fpus * fpuops * (vectorlen / (8 * 64)) * turbofreq,
	}
}

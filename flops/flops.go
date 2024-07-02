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

	cores := float64(info.PhysicalCores)

	// This implementation is not perfect;
	// Usually the throughput of FMA on Intel's processor is 0.5, so defining it here as 0.5.
	fmaReciprocalThroughput := 0.5

	instpercycle := float64(1)

	if info.Supported.FMA {
		instpercycle = 2 // if FMA is supported, it can compute both ADD and MUL in 1 op.

		// consider FMA inst throughput.
		// Because throughput is reciprocal, division is required.
		instpercycle *= 1 / fmaReciprocalThroughput
	}

	instpervectorFloat := float64(1)
	instpervectorDouble := float64(1)
	switch {
	case info.Supported.AVX2, info.Supported.AVX:
		instpervectorFloat = 256 / 32
		instpervectorDouble = 256 / 64

	case info.Supported.SSE2, info.Supported.SSE:
		instpervectorFloat = 128 / 32
		instpervectorDouble = 128 / 64

	case info.Supported.MMX:
		instpervectorFloat = 64 / 32
		instpervectorDouble = 64 / 64
	}

	instpercycleFloat := instpercycle * instpervectorFloat
	instpercycleDouble := instpercycle * instpervectorDouble

	return &Flops{
		MFlopsFloatBase:   cores * instpercycleFloat * basefreq,
		MFlopsFloatTurbo:  cores * instpercycleFloat * turbofreq,
		MFlopsDoubleBase:  cores * instpercycleDouble * basefreq,
		MFlopsDoubleTurbo: cores * instpercycleDouble * turbofreq,
	}
}

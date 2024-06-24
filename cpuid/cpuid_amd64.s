//+build amd64

// func(op uint32) (eax, ebx, ecx, edx uint32)
TEXT ·asmcpuid(SB),$0-24
	XORQ CX, CX		// CX = 0 just in case
	MOVL op+0(FP), AX	// AX = op
	CPUID
	MOVL AX, eax+8(FP)	// eax = AX
	MOVL BX, ebx+12(FP)	// ebx = BX
	MOVL CX, ecx+16(FP)	// ecx = CX
	MOVL DX, edx+20(FP)	// edx = DX
	RET

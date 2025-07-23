//+build amd64

// func(leaf uint32) (eax, ebx, ecx, edx uint32)
TEXT ·asmcpuid1(SB),$0-24
	XORQ CX, CX		// CX = 0 just in case
	MOVL leaf+0(FP), AX	// AX = leaf
	CPUID
	MOVL AX, eax+8(FP)	// eax = AX
	MOVL BX, ebx+12(FP)	// ebx = BX
	MOVL CX, ecx+16(FP)	// ecx = CX
	MOVL DX, edx+20(FP)	// edx = DX
	RET

// func(leaf, subleaf uint32) (eax, ebx, ecx, edx uint32)
TEXT ·asmcpuid2(SB),$0-24
	XORQ CX, CX		// CX = 0 just in case
	MOVL leaf+0(FP), AX	// AX = leaf
	MOVL subleaf+4(FP), CX	// CX = subleaf
	CPUID
	MOVL AX, eax+8(FP)	// eax = AX
	MOVL BX, ebx+12(FP)	// ebx = BX
	MOVL CX, ecx+16(FP)	// ecx = CX
	MOVL DX, edx+20(FP)	// edx = DX
	RET

// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm

#include "textflag.h"

// func SaxpyUnitary(alpha float32, x, y []float32)
TEXT ·SaxpyUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), SI  // SI = &x
	MOVQ    y_base+32(FP), DI // DI = &y
	MOVQ    x_len+16(FP), BX  // BX = min( len(x), len(y) )
	CMPQ    y_len+40(FP), BX
	CMOVQLE y_len+40(FP), BX
	CMPQ    BX, $0            // if BX == 0 { return }
	JE      saxpy_end
	MOVSS   alpha+0(FP), X0
	SHUFPS  $0, X0, X0        // X0 = { a, a, a, a }
	XORQ    AX, AX            // i = 0
	PXOR    X2, X2            // 2 NOP instructions (PXOR) to align
	PXOR    X3, X3            // loop to cache line
	MOVQ    DI, CX
	ANDQ    $0xF, CX          // Align on 16-byte boundary for ADDPS
	JZ      saxpy_no_trim      // if CX == 0 { goto saxpy_no_trim }

	XORQ $0xF, CX // CX = 4 - floor( BX % 16 / 4 )
	INCQ CX
	SHRQ $2, CX

saxpy_align: // Trim first value(s) in unaligned buffer  do {
	MOVSS (SI)(AX*4), X2 // X2 = x[i]
	MULSS X0, X2         // X2 *= a
	ADDSS (DI)(AX*4), X2 // X2 += y[i]
	MOVSS X2, (DI)(AX*4) // y[i] = X2
	INCQ  AX             // i++
	DECQ  BX
	JZ    saxpy_end       // if --BX == 0 { return }
	LOOP  saxpy_align     // } while --CX > 0

saxpy_no_trim:
	MOVUPS X0, X1           // Copy X0 to X1 for pipelining
	MOVQ   BX, CX
	ANDQ   $0xF, BX         // BX = len % 16
	SHRQ   $4, CX           // CX = int( len / 16 )
	JZ     saxpy_tail4_start // if CX == 0 { return }

saxpy_loop: // Loop unrolled 16x   do {
	MOVUPS (SI)(AX*4), X2   // X2 = x[i:i+4]
	MOVUPS 16(SI)(AX*4), X3
	MOVUPS 32(SI)(AX*4), X4
	MOVUPS 48(SI)(AX*4), X5
	MULPS  X0, X2           // X2 *= a
	MULPS  X1, X3
	MULPS  X0, X4
	MULPS  X1, X5
	ADDPS  (DI)(AX*4), X2   // X2 += y[i:i+4]
	ADDPS  16(DI)(AX*4), X3
	ADDPS  32(DI)(AX*4), X4
	ADDPS  48(DI)(AX*4), X5
	MOVUPS X2, (DI)(AX*4)   // dst[i:i+4] = X2
	MOVUPS X3, 16(DI)(AX*4)
	MOVUPS X4, 32(DI)(AX*4)
	MOVUPS X5, 48(DI)(AX*4)
	ADDQ   $16, AX          // i += 16
	LOOP   saxpy_loop        // while (--CX) > 0
	CMPQ   BX, $0           // if BX == 0 { return }
	JE     saxpy_end

saxpy_tail4_start: // Reset loop counter for 4-wide tail loop
	MOVQ BX, CX          // CX = floor( BX / 4 )
	SHRQ $2, CX
	JZ   saxpy_tail_start // if CX == 0 { goto saxpy_tail_start }

saxpy_tail4: // Loop unrolled 4x   do {
	MOVUPS (SI)(AX*4), X2 // X2 = x[i]
	MULPS  X0, X2         // X2 *= a
	ADDPS  (DI)(AX*4), X2 // X2 += y[i]
	MOVUPS X2, (DI)(AX*4) // y[i] = X2
	ADDQ   $4, AX         // i += 4
	LOOP   saxpy_tail4     // } while --CX > 0

saxpy_tail_start: // Reset loop counter for 1-wide tail loop
	MOVQ BX, CX   // CX = BX % 4
	ANDQ $3, CX
	JZ   saxpy_end // if CX == 0 { return }

saxpy_tail:
	MOVSS (SI)(AX*4), X1 // X1 = x[i]
	MULSS X0, X1         // X1 *= a
	ADDSS (DI)(AX*4), X1 // X1 += y[i]
	MOVSS X1, (DI)(AX*4) // y[i] = X1
	INCQ  AX             // i++
	LOOP  saxpy_tail      // } while --CX > 0

saxpy_end:
	RET

// func SaxpyInc(alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)
TEXT ·SaxpyInc(SB), NOSPLIT, $0
	MOVQ  n+56(FP), CX      // CX = n
	CMPQ  CX, $0            // if n==0 { return }
	JLE   saxpyi_end
	MOVQ  x_base+8(FP), SI  // SI = &x
	MOVQ  y_base+32(FP), DI // DI = &y
	MOVQ  ix+80(FP), R8     // R8 = ix
	MOVQ  iy+88(FP), R9     // R9 = iy
	LEAQ  (SI)(R8*4), SI    // SI = &(x[ix])
	LEAQ  (DI)(R9*4), DI    // DI = &(y[iy])
	MOVQ  DI, DX            // DX = DI   Read Pointer for y
	MOVQ  incX+64(FP), R8   // R8 = incX
	SHLQ  $2, R8            // R8 *= sizeof(float32)
	MOVQ  incY+72(FP), R9   // R9 = incY
	SHLQ  $2, R9            // R9 *= sizeof(float32)
	MOVSS alpha+0(FP), X0   // X0 = alpha
	MOVSS X0, X1            // X1 = X0  // for pipelining
	MOVQ  CX, BX
	ANDQ  $3, BX            // BX = n % 4
	SHRQ  $2, CX            // CX = floor( n / 4 )
	JZ    saxpyi_tail_start  // if CX == 0 { goto saxpyi_tail_start }

saxpyi_loop: // Loop unrolled 4x   do {
	MOVSS (SI), X2       // X_i = x[i]
	MOVSS (SI)(R8*1), X3
	LEAQ  (SI)(R8*2), SI // SI = &(SI[incX*2])
	MOVSS (SI), X4
	MOVSS (SI)(R8*1), X5
	MULSS X1, X2         // X_i *= a
	MULSS X0, X3
	MULSS X1, X4
	MULSS X0, X5
	ADDSS (DX), X2       // X_i += y[i]
	ADDSS (DX)(R9*1), X3
	LEAQ  (DX)(R9*2), DX // DX = &(DX[incY*2])
	ADDSS (DX), X4
	ADDSS (DX)(R9*1), X5
	MOVSS X2, (DI)       // y[i] = X_i
	MOVSS X3, (DI)(R9*1)
	LEAQ  (DI)(R9*2), DI // DI = &(DI[incY*2])
	MOVSS X4, (DI)
	MOVSS X5, (DI)(R9*1)
	LEAQ  (SI)(R8*2), SI // SI = &(SI[incX*2])  // Increment addresses
	LEAQ  (DX)(R9*2), DX // DX = &(DX[incY*2])
	LEAQ  (DI)(R9*2), DI // DI = &(DI[incY*2])
	LOOP  saxpyi_loop     // } while --CX > 0
	CMPQ  BX, $0         // if BX == 0 { return }
	JE    saxpyi_end

saxpyi_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX

saxpyi_tail: // do {
	MOVSS (SI), X2   // X2 = x[i]
	MULSS X1, X2     // X2 *= a
	ADDSS (DI), X2   // X2 += y[i]
	MOVSS X2, (DI)   // y[i] = X2
	ADDQ  R8, SI     // SI = &(SI[incX])
	ADDQ  R9, DI     // DI = &(DI[incY])
	LOOP  saxpyi_tail // } while --CX > 0

saxpyi_end:
	RET

#define HADDPS_SUM_SUM    LONG $0xC07C0FF2 // @ HADDPS X0, X0

#define X_PTR SI
#define Y_PTR DI
#define LEN CX
#define TAIL BX
#define IDX AX
#define SUM X0
#define P_SUM X1

// func DotUnitary(x, y []float32) (sum float32)
TEXT ·DotUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ    y_base+24(FP), Y_PTR // Y_PTR = &y
	PXOR    SUM, SUM             // SUM = 0
	MOVQ    x_len+8(FP), LEN     // LEN = min( len(x), len(y) )
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	CMPQ    LEN, $0
	JE      dot_end

	XORQ IDX, IDX
	MOVQ Y_PTR, DX
	ANDQ $0xF, DX    // Align on 16-byte boundary for MULPS
	JZ   dot_no_trim // if DX == 0 { goto dot_no_trim }
	SUBQ $16, DX

dot_align: // Trim first value(s) in unaligned buffer  do {
	MOVSS (X_PTR)(IDX*4), X2 // X2 = x[i]
	MULSS (Y_PTR)(IDX*4), X2 // X2 *= y[i]
	ADDSS X2, SUM            // SUM += X2
	INCQ  IDX                // IDX++
	DECQ  LEN
	JZ    dot_end            // if --TAIL == 0 { return }
	ADDQ  $4, DX
	JNZ   dot_align          // } while --DX > 0

dot_no_trim:
	PXOR P_SUM, P_SUM    // P_SUM = 0  for pipelining
	MOVQ LEN, TAIL
	ANDQ $0xF, TAIL      // TAIL = LEN % 16
	SHRQ $4, LEN         // LEN = floor( LEN / 16 )
	JZ   dot_tail4_start // if LEN == 0 { goto dot_tail4_start }

dot_loop: // Loop unrolled 16x  do {
	MOVUPS (X_PTR)(IDX*4), X2   // X_i = x[i:i+1]
	MOVUPS 16(X_PTR)(IDX*4), X3
	MOVUPS 32(X_PTR)(IDX*4), X4
	MOVUPS 48(X_PTR)(IDX*4), X5

	MULPS (Y_PTR)(IDX*4), X2   // X_i *= y[i:i+1]
	MULPS 16(Y_PTR)(IDX*4), X3
	MULPS 32(Y_PTR)(IDX*4), X4
	MULPS 48(Y_PTR)(IDX*4), X5

	ADDPS X2, SUM   // SUM += X_i
	ADDPS X3, P_SUM
	ADDPS X4, SUM
	ADDPS X5, P_SUM

	ADDQ $16, IDX // IDX += 16
	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDPS P_SUM, SUM // SUM += P_SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail4_start: // Reset loop counter for 4-wide tail loop
	MOVQ TAIL, LEN      // LEN = floor( TAIL / 4 )
	SHRQ $2, LEN
	JZ   dot_tail_start // if LEN == 0 { goto dot_tail_start }

dot_tail4_loop: // Loop unrolled 4x  do {
	MOVUPS (X_PTR)(IDX*4), X2 // X_i = x[i:i+1]
	MULPS  (Y_PTR)(IDX*4), X2 // X_i *= y[i:i+1]
	ADDPS  X2, SUM            // SUM += X_i
	ADDQ   $4, IDX            // i += 4
	DECQ   LEN
	JNZ    dot_tail4_loop     // } while --LEN > 0

dot_tail_start: // Reset loop counter for 1-wide tail loop
	ANDQ $3, TAIL // TAIL = TAIL % 4
	JZ   dot_end  // if TAIL == 0 { return }

dot_tail: // do {
	MOVSS (X_PTR)(IDX*4), X2 // X2 = x[i]
	MULSS (Y_PTR)(IDX*4), X2 // X2 *= y[i]
	ADDSS X2, SUM            // psum += X2
	INCQ  IDX                // IDX++
	DECQ  TAIL
	JNZ   dot_tail           // } while --TAIL > 0

dot_end:
	HADDPS_SUM_SUM        // SUM = \sum{ SUM[i] }
	HADDPS_SUM_SUM
	MOVSS SUM, sum+48(FP) // return SUM
	RET

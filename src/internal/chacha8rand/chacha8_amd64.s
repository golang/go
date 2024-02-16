// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// ChaCha8 is ChaCha with 8 rounds.
// See https://cr.yp.to/chacha/chacha-20080128.pdf.
// See chacha8_generic.go for additional details.

// ROL rotates the uint32s in register R left by N bits, using temporary T.
#define ROL(N, R, T) \
	MOVO R, T; PSLLL $(N), T; PSRLL $(32-(N)), R; PXOR T, R

// ROL16 rotates the uint32s in register R left by 16, using temporary T if needed.
#ifdef GOAMD64_v2
#define ROL16(R, T) PSHUFB ·rol16<>(SB), R
#else
#define ROL16(R, T) ROL(16, R, T)
#endif

// ROL8 rotates the uint32s in register R left by 8, using temporary T if needed.
#ifdef GOAMD64_v2
#define ROL8(R, T) PSHUFB ·rol8<>(SB), R
#else
#define ROL8(R, T) ROL(8, R, T)
#endif

// QR is the ChaCha quarter-round on A, B, C, and D. T is an available temporary.
#define QR(A, B, C, D, T) \
	PADDD B, A; PXOR A, D; ROL16(D, T); \
	PADDD D, C; PXOR C, B; MOVO B, T; PSLLL $12, T; PSRLL $20, B; PXOR T, B; \
	PADDD B, A; PXOR A, D; ROL8(D, T); \
	PADDD D, C; PXOR C, B; MOVO B, T; PSLLL $7, T; PSRLL $25, B; PXOR T, B

// REPLREG replicates the register R into 4 uint32s in XR.
#define REPLREG(R, XR) \
	MOVQ R, XR; \
	PSHUFD $0, XR, XR

// REPL replicates the uint32 constant val into 4 uint32s in XR. It smashes DX.
#define REPL(val, XR) \
	MOVL $val, DX; \
	REPLREG(DX, XR)

// SEED copies the off'th uint32 of the seed into the register XR,
// replicating it into all four stripes of the register.
#define SEED(off, reg, XR) \
	MOVL (4*off)(AX), reg; \
	REPLREG(reg, XR) \

// block runs 4 ChaCha8 block transformations in the four stripes of the X registers.

// func block(seed *[8]uint32, blocks *[16][4]uint32, counter uint32)
TEXT ·block<ABIInternal>(SB), NOSPLIT, $16
	// seed in AX
	// blocks in BX
	// counter in CX

	// Load initial constants into top row.
	REPL(0x61707865, X0)
	REPL(0x3320646e, X1)
	REPL(0x79622d32, X2)
	REPL(0x6b206574, X3)

	// Load counter into bottom left cell.
	// Each stripe gets a different counter: 0, 1, 2, 3.
	// (PINSRD is not available in GOAMD64_v1,
	// so just do it in memory on all systems.
	// This is not on the critical path.)
	MOVL CX, 0(SP)
	INCL CX
	MOVL CX, 4(SP)
	INCL CX
	MOVL CX, 8(SP)
	INCL CX
	MOVL CX, 12(SP)
	MOVOU 0(SP), X12

	// Load seed words into next two rows and into DI, SI, R8..R13
	SEED(0, DI, X4)
	SEED(1, SI, X5)
	SEED(2, R8, X6)
	SEED(3, R9, X7)
	SEED(4, R10, X8)
	SEED(5, R11, X9)
	SEED(6, R12, X10)
	SEED(7, R13, X11)

	// Zeros for remaining two matrix entries.
	// We have just enough XMM registers to hold the state,
	// without one for the temporary, so we flush and restore
	// some values to and from memory to provide a temporary.
	// The initial temporary is X15, so zero its memory instead
	// of X15 itself.
	MOVL $0, DX
	MOVQ DX, X13
	MOVQ DX, X14
	MOVOU X14, (15*16)(BX)

	// 4 iterations. Each iteration is 8 quarter-rounds.
	MOVL $4, DX
loop:
	QR(X0, X4, X8, X12, X15)
	MOVOU X4, (4*16)(BX) // save X4
	QR(X1, X5, X9, X13, X15)
	MOVOU (15*16)(BX), X15 // reload X15; temp now X4
	QR(X2, X6, X10, X14, X4)
	QR(X3, X7, X11, X15, X4)

	QR(X0, X5, X10, X15, X4)
	MOVOU X15, (15*16)(BX) // save X15
	QR(X1, X6, X11, X12, X4)
	MOVOU (4*16)(BX), X4  // reload X4; temp now X15
	QR(X2, X7, X8, X13, X15)
	QR(X3, X4, X9, X14, X15)

	DECL DX
	JNZ loop

	// Store interlaced blocks back to output buffer,
	// adding original seed along the way.

	// First the top and bottom rows.
	MOVOU X0, (0*16)(BX)
	MOVOU X1, (1*16)(BX)
	MOVOU X2, (2*16)(BX)
	MOVOU X3, (3*16)(BX)
	MOVOU X12, (12*16)(BX)
	MOVOU X13, (13*16)(BX)
	MOVOU X14, (14*16)(BX)
	// X15 has already been stored.

	// Now we have X0-X3, X12-X15 available for temporaries.
	// Add seed rows back to output. We left seed in DI, SI, R8..R13 above.
	REPLREG(DI, X0)
	REPLREG(SI, X1)
	REPLREG(R8, X2)
	REPLREG(R9, X3)
	REPLREG(R10, X12)
	REPLREG(R11, X13)
	REPLREG(R12, X14)
	REPLREG(R13, X15)
	PADDD X0, X4
	PADDD X1, X5
	PADDD X2, X6
	PADDD X3, X7
	PADDD X12, X8
	PADDD X13, X9
	PADDD X14, X10
	PADDD X15, X11
	MOVOU X4, (4*16)(BX)
	MOVOU X5, (5*16)(BX)
	MOVOU X6, (6*16)(BX)
	MOVOU X7, (7*16)(BX)
	MOVOU X8, (8*16)(BX)
	MOVOU X9, (9*16)(BX)
	MOVOU X10, (10*16)(BX)
	MOVOU X11, (11*16)(BX)

	MOVL $0, AX
	MOVQ AX, X15 // must be 0 on return

	RET

// rotate left 16 indexes for PSHUFB
GLOBL ·rol16<>(SB), NOPTR|RODATA, $16
DATA ·rol16<>+0(SB)/8, $0x0504070601000302
DATA ·rol16<>+8(SB)/8, $0x0D0C0F0E09080B0A

// rotate left 8 indexes for PSHUFB
GLOBL ·rol8<>(SB), NOPTR|RODATA, $16
DATA ·rol8<>+0(SB)/8, $0x0605040702010003
DATA ·rol8<>+8(SB)/8, $0x0E0D0C0F0A09080B

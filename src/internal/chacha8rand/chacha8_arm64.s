// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// QR is the ChaCha quarter-round on A, B, C, and D.
// V30 is used as a temporary, and V31 is assumed to
// hold the index table for rotate left 8.
#define QR(A, B, C, D) \
	VADD A.S4, B.S4, A.S4; VEOR D.B16, A.B16, D.B16;   VREV32 D.H8, D.H8; \
	VADD C.S4, D.S4, C.S4; VEOR B.B16, C.B16, V30.B16; VSHL $12, V30.S4, B.S4; VSRI $20, V30.S4, B.S4 \
	VADD A.S4, B.S4, A.S4; VEOR D.B16, A.B16, D.B16;   VTBL V31.B16, [D.B16], D.B16; \
	VADD C.S4, D.S4, C.S4; VEOR B.B16, C.B16, V30.B16; VSHL  $7, V30.S4, B.S4; VSRI $25, V30.S4, B.S4

// block runs 4 ChaCha8 block transformations in the four stripes of the V registers.

// func block(seed *[8]uint32, blocks *[4][16]uint32, counter uint32)
TEXT ·block<ABIInternal>(SB), NOSPLIT, $16
	// seed in R0
	// blocks in R1
	// counter in R2

	// Load initial constants into top row.
	MOVD $·chachaConst(SB), R10
	VLD4R (R10), [V0.S4, V1.S4, V2.S4, V3.S4]

	// Load increment and rotate 8 constants into V30, V31.
	MOVD $·chachaIncRot(SB), R11
	VLD1 (R11), [V30.S4, V31.S4]

	VLD4R.P 16(R0), [V4.S4, V5.S4, V6.S4, V7.S4]
	VLD4R.P 16(R0), [V8.S4, V9.S4, V10.S4, V11.S4]

	// store counter to memory to replicate its uint32 halfs back out
	MOVW R2, 0(RSP)
	VLD1R 0(RSP), [V12.S4]

	// Add 0, 1, 2, 3 to counter stripes.
	VADD V30.S4, V12.S4, V12.S4

	// Zeros for remaining two matrix entries.
	VEOR V13.B16, V13.B16, V13.B16
	VEOR V14.B16, V14.B16, V14.B16
	VEOR V15.B16, V15.B16, V15.B16

	// Save seed state for adding back later.
	VMOV V4.B16, V20.B16
	VMOV V5.B16, V21.B16
	VMOV V6.B16, V22.B16
	VMOV V7.B16, V23.B16
	VMOV V8.B16, V24.B16
	VMOV V9.B16, V25.B16
	VMOV V10.B16, V26.B16
	VMOV V11.B16, V27.B16

	// 4 iterations. Each iteration is 8 quarter-rounds.
	MOVD $4, R0
loop:
	QR(V0, V4, V8, V12)
	QR(V1, V5, V9, V13)
	QR(V2, V6, V10, V14)
	QR(V3, V7, V11, V15)

	QR(V0, V5, V10, V15)
	QR(V1, V6, V11, V12)
	QR(V2, V7, V8, V13)
	QR(V3, V4, V9, V14)

	SUB $1, R0
	CBNZ R0, loop

	// Add seed back.
	VADD V4.S4, V20.S4, V4.S4
	VADD V5.S4, V21.S4, V5.S4
	VADD V6.S4, V22.S4, V6.S4
	VADD V7.S4, V23.S4, V7.S4
	VADD V8.S4, V24.S4, V8.S4
	VADD V9.S4, V25.S4, V9.S4
	VADD V10.S4, V26.S4, V10.S4
	VADD V11.S4, V27.S4, V11.S4

	// Store interlaced blocks back to output buffer.
	VST1.P [ V0.B16,  V1.B16,  V2.B16,  V3.B16], 64(R1)
	VST1.P [ V4.B16,  V5.B16,  V6.B16,  V7.B16], 64(R1)
	VST1.P [ V8.B16,  V9.B16, V10.B16, V11.B16], 64(R1)
	VST1.P [V12.B16, V13.B16, V14.B16, V15.B16], 64(R1)
	RET

GLOBL	·chachaConst(SB), NOPTR|RODATA, $32
DATA	·chachaConst+0x00(SB)/4, $0x61707865
DATA	·chachaConst+0x04(SB)/4, $0x3320646e
DATA	·chachaConst+0x08(SB)/4, $0x79622d32
DATA	·chachaConst+0x0c(SB)/4, $0x6b206574

GLOBL	·chachaIncRot(SB), NOPTR|RODATA, $32
DATA	·chachaIncRot+0x00(SB)/4, $0x00000000
DATA	·chachaIncRot+0x04(SB)/4, $0x00000001
DATA	·chachaIncRot+0x08(SB)/4, $0x00000002
DATA	·chachaIncRot+0x0c(SB)/4, $0x00000003
DATA	·chachaIncRot+0x10(SB)/4, $0x02010003
DATA	·chachaIncRot+0x14(SB)/4, $0x06050407
DATA	·chachaIncRot+0x18(SB)/4, $0x0A09080B
DATA	·chachaIncRot+0x1c(SB)/4, $0x0E0D0C0F

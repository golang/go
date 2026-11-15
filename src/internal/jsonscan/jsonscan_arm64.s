// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Register use in every kernel:
//   R0: p, start of the input
//   R1: n, input length, n >= 16
//   R2: start of the current block
//   R3: start of the last block, p+n-16
//   R4: hit syndrome of the current block, bits 4i to 4i+3 for byte R2+i
//   V0: current block; V1 to V3: scratch
//   V20: 0x20; V21: '"'; V22: '\\'; V23: '<'; V24: '>'; V25: '&'
// Loop invariant: no byte in [R0, R2) is special and R2 < R3, so the block
// load stays inside [R0, R0+R1). The last block is loaded from R3, which may
// overlap bytes already found clean but ends exactly at R0+R1.
// A signed compare 0x20 > c holds for c < 0x20 and for c >= 0x80.

#define SPECIAL_NEON() \
	VCMGT	V0.B16, V20.B16, V1.B16; \
	VCMEQ	V0.B16, V21.B16, V2.B16; \
	VCMEQ	V0.B16, V22.B16, V3.B16; \
	VORR	V2.B16, V1.B16, V1.B16; \
	VORR	V3.B16, V1.B16, V1.B16

#define HTML_NEON() \
	SPECIAL_NEON(); \
	VCMEQ	V0.B16, V23.B16, V2.B16; \
	VCMEQ	V0.B16, V24.B16, V3.B16; \
	VORR	V2.B16, V1.B16, V1.B16; \
	VORR	V3.B16, V1.B16, V1.B16; \
	VCMEQ	V0.B16, V25.B16, V2.B16; \
	VORR	V2.B16, V1.B16, V1.B16

#define SYNDROME() \
	VSHRN	$4, V1.H8, V1.B8; \
	VMOV	V1.D[0], R4

// func indexStringSpecial(p *byte, n int) int
TEXT ·indexStringSpecial(SB), NOSPLIT, $0-24
	MOVD	p+0(FP), R0
	MOVD	n+8(FP), R1
	VMOVI	$0x20, V20.B16
	VMOVI	$0x22, V21.B16
	VMOVI	$0x5c, V22.B16
	MOVD	R0, R2
	ADD	R0, R1, R3
	SUB	$16, R3, R3
	B	special_next

special_loop:
	VLD1	(R2), [V0.B16]
	SPECIAL_NEON()
	SYNDROME()
	CBNZ	R4, special_hit
	ADD	$16, R2, R2

special_next:
	CMP	R3, R2
	BLO	special_loop
	MOVD	R3, R2
	VLD1	(R2), [V0.B16]
	SPECIAL_NEON()
	SYNDROME()
	CBNZ	R4, special_hit
	MOVD	R1, ret+16(FP)
	RET

special_hit:
	RBIT	R4, R4
	CLZ	R4, R4
	SUB	R0, R2, R2
	ADD	R4>>2, R2, R2
	MOVD	R2, ret+16(FP)
	RET

// func indexEscapeHTML(p *byte, n int) int
TEXT ·indexEscapeHTML(SB), NOSPLIT, $0-24
	MOVD	p+0(FP), R0
	MOVD	n+8(FP), R1
	VMOVI	$0x20, V20.B16
	VMOVI	$0x22, V21.B16
	VMOVI	$0x5c, V22.B16
	VMOVI	$0x3c, V23.B16
	VMOVI	$0x3e, V24.B16
	VMOVI	$0x26, V25.B16
	MOVD	R0, R2
	ADD	R0, R1, R3
	SUB	$16, R3, R3
	B	html_next

html_loop:
	VLD1	(R2), [V0.B16]
	HTML_NEON()
	SYNDROME()
	CBNZ	R4, html_hit
	ADD	$16, R2, R2

html_next:
	CMP	R3, R2
	BLO	html_loop
	MOVD	R3, R2
	VLD1	(R2), [V0.B16]
	HTML_NEON()
	SYNDROME()
	CBNZ	R4, html_hit
	MOVD	R1, ret+16(FP)
	RET

html_hit:
	RBIT	R4, R4
	CLZ	R4, R4
	SUB	R0, R2, R2
	ADD	R4>>2, R2, R2
	MOVD	R2, ret+16(FP)
	RET

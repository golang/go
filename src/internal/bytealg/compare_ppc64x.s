// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "textflag.h"

// Helper names for x-form loads in BE ordering.
#ifdef  GOARCH_ppc64le
#define _LDBEX	MOVDBR
#define _LWBEX	MOVWBR
#define _LHBEX	MOVHBR
#else
#define _LDBEX	MOVD
#define _LWBEX	MOVW
#define _LHBEX	MOVH
#endif

#ifdef GOPPC64_power9
#define SETB_CR0(rout) SETB CR0, rout
#define SETB_CR1(rout) SETB CR1, rout
#define SETB_INIT()
#define SETB_CR0_NE(rout) SETB_CR0(rout)
#else
// A helper macro to emulate SETB on P8. This assumes
// -1 is in R20, and 1 is in R21. crxlt and crxeq must
// also be the same CR field.
#define _SETB(crxlt, crxeq, rout) \
	ISEL	crxeq,R0,R21,rout \
	ISEL	crxlt,R20,rout,rout

// A special case when it is know the comparison
// will always be not equal. The result must be -1 or 1.
#define SETB_CR0_NE(rout) \
	ISEL	CR0LT,R20,R21,rout

#define SETB_CR0(rout) _SETB(CR0LT, CR0EQ, rout)
#define SETB_CR1(rout) _SETB(CR1LT, CR1EQ, rout)
#define SETB_INIT() \
	MOVD	$-1,R20 \
	MOVD	$1,R21
#endif

TEXT ·Compare<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
	// incoming:
	// R3 a addr
	// R4 a len
	// R6 b addr
	// R7 b len
	//
	// on entry to cmpbody:
	// R3 return value if len(a) == len(b)
	// R5 a addr
	// R6 b addr
	// R9 min(len(a),len(b))
	SETB_INIT()
	MOVD	R3,R5
	CMP	R4,R7,CR0
	CMP	R3,R6,CR7
	ISEL	CR0LT,R4,R7,R9
	SETB_CR0(R3)
	BC	$12,30,LR	// beqlr cr7
	BR	cmpbody<>(SB)

TEXT runtime·cmpstring<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
	// incoming:
	// R3 a addr -> R5
	// R4 a len  -> R3
	// R5 b addr -> R6
	// R6 b len  -> R4
	//
	// on entry to cmpbody:
	// R3 compare value if compared length is same.
	// R5 a addr
	// R6 b addr
	// R9 min(len(a),len(b))
	SETB_INIT()
	CMP	R4,R6,CR0
	CMP	R3,R5,CR7
	ISEL	CR0LT,R4,R6,R9
	MOVD	R5,R6
	MOVD	R3,R5
	SETB_CR0(R3)
	BC	$12,30,LR	// beqlr cr7
	BR	cmpbody<>(SB)

#ifdef GOARCH_ppc64le
DATA byteswap<>+0(SB)/8, $0x0706050403020100
DATA byteswap<>+8(SB)/8, $0x0f0e0d0c0b0a0908
GLOBL byteswap<>+0(SB), RODATA, $16
#define SWAP V21
#endif

TEXT cmpbody<>(SB),NOSPLIT|NOFRAME,$0-0
start:
	CMP	R9,$16,CR0
	CMP	R9,$32,CR1
	CMP	R9,$64,CR2
	MOVD	$16,R10
	BLT	cmp8
	BLT	CR1,cmp16
	BLT	CR2,cmp32

cmp64:	// >= 64B
	DCBT	(R5)		// optimize for size>=64
	DCBT	(R6)		// cache hint

	SRD	$6,R9,R14	// There is at least one iteration.
	MOVD	R14,CTR
	ANDCC   $63,R9,R9
	CMP	R9,$16,CR1	// Do setup for tail check early on.
	CMP	R9,$32,CR2
	CMP	R9,$48,CR3
	ADD	$-16,R9,R9

	MOVD	$32,R11		// set offsets to load into vector
	MOVD	$48,R12		// set offsets to load into vector

	PCALIGN	$16
cmp64_loop:
	LXVD2X	(R5)(R0),V3	// load bytes of A at offset 0 into vector
	LXVD2X	(R6)(R0),V4	// load bytes of B at offset 0 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different	// jump out if its different

	LXVD2X	(R5)(R10),V3	// load bytes of A at offset 16 into vector
	LXVD2X	(R6)(R10),V4	// load bytes of B at offset 16 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R11),V3	// load bytes of A at offset 32 into vector
	LXVD2X	(R6)(R11),V4	// load bytes of B at offset 32 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R12),V3	// load bytes of A at offset 64 into vector
	LXVD2X	(R6)(R12),V4	// load bytes of B at offset 64 into vector
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	ADD	$64,R5,R5	// increment to next 64 bytes of A
	ADD	$64,R6,R6	// increment to next 64 bytes of B
	BDNZ	cmp64_loop
	BC	$12,2,LR	// beqlr

	// Finish out tail with minimal overlapped checking.
	// Note, 0 tail is handled by beqlr above.
	BLE	CR1,cmp64_tail_gt0
	BLE	CR2,cmp64_tail_gt16
	BLE	CR3,cmp64_tail_gt32

cmp64_tail_gt48: // 49 - 63 B
	LXVD2X	(R0)(R5),V3
	LXVD2X	(R0)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R10),V3
	LXVD2X	(R6)(R10),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R11),V3
	LXVD2X	(R6)(R11),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	BR cmp64_tail_gt0

	PCALIGN $16
cmp64_tail_gt32: // 33 - 48B
	LXVD2X	(R0)(R5),V3
	LXVD2X	(R0)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R5)(R10),V3
	LXVD2X	(R6)(R10),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	BR cmp64_tail_gt0

	PCALIGN $16
cmp64_tail_gt16: // 17 - 32B
	LXVD2X	(R0)(R5),V3
	LXVD2X	(R0)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	BR cmp64_tail_gt0

	PCALIGN $16
cmp64_tail_gt0: // 1 - 16B
	LXVD2X	(R5)(R9),V3
	LXVD2X	(R6)(R9),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	RET

	PCALIGN $16
cmp32:	// 32 - 63B
	ANDCC	$31,R9,R9

	LXVD2X	(R0)(R5),V3
	LXVD2X	(R0)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R10)(R5),V3
	LXVD2X	(R10)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	BC	$12,2,LR	// beqlr
	ADD	R9,R10,R10

	LXVD2X	(R9)(R5),V3
	LXVD2X	(R9)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different

	LXVD2X	(R10)(R5),V3
	LXVD2X	(R10)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different
	RET

	PCALIGN $16
cmp16:	// 16 - 31B
	ANDCC	$15,R9,R9
	LXVD2X	(R0)(R5),V3
	LXVD2X	(R0)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different
	BC	$12,2,LR	// beqlr

	LXVD2X	(R9)(R5),V3
	LXVD2X	(R9)(R6),V4
	VCMPEQUDCC	V3,V4,V1
	BGE	CR6,different
	RET

	PCALIGN $16
different:
#ifdef	GOARCH_ppc64le
	MOVD	$byteswap<>+00(SB),R16
	LXVD2X	(R16)(R0),SWAP	// Set up swap string

	VPERM	V3,V3,SWAP,V3
	VPERM	V4,V4,SWAP,V4
#endif

	MFVSRD	VS35,R16	// move upper doublewords of A and B into GPR for comparison
	MFVSRD	VS36,R10

	CMPU	R16,R10
	BEQ	lower
	SETB_CR0_NE(R3)
	RET

	PCALIGN $16
lower:
	VSLDOI	$8,V3,V3,V3	// move lower doublewords of A and B into GPR for comparison
	MFVSRD	VS35,R16
	VSLDOI	$8,V4,V4,V4
	MFVSRD	VS36,R10

	CMPU	R16,R10
	SETB_CR0_NE(R3)
	RET

	PCALIGN $16
cmp8:	// 8 - 15B (0 - 15B if GOPPC64_power10)
#ifdef GOPPC64_power10
	SLD	$56,R9,R9
	LXVLL	R5,R9,V3	// Load bytes starting from MSB to LSB, unused are zero filled.
	LXVLL	R6,R9,V4
	VCMPUQ	V3,V4,CR0	// Compare as a 128b integer.
	SETB_CR0(R6)
	ISEL	CR0EQ,R3,R6,R3	// If equal, length determines the return value.
	RET
#else
	CMP	R9,$8
	BLT	cmp4
	ANDCC	$7,R9,R9
	_LDBEX	(R0)(R5),R10
	_LDBEX	(R0)(R6),R11
	_LDBEX	(R9)(R5),R12
	_LDBEX	(R9)(R6),R14
	CMPU	R10,R11,CR0
	SETB_CR0(R5)
	CMPU	R12,R14,CR1
	SETB_CR1(R6)
	CRAND   CR0EQ,CR1EQ,CR1EQ // If both equal, length determines return value.
	ISEL	CR0EQ,R6,R5,R4
	ISEL	CR1EQ,R3,R4,R3
	RET

	PCALIGN	$16
cmp4:	// 4 - 7B
	CMP	R9,$4
	BLT	cmp2
	ANDCC	$3,R9,R9
	_LWBEX	(R0)(R5),R10
	_LWBEX	(R0)(R6),R11
	_LWBEX	(R9)(R5),R12
	_LWBEX	(R9)(R6),R14
	RLDIMI	$32,R10,$0,R12
	RLDIMI	$32,R11,$0,R14
	CMPU	R12,R14
	BR	cmp0

	PCALIGN $16
cmp2:	// 2 - 3B
	CMP	R9,$2
	BLT	cmp1
	ANDCC	$1,R9,R9
	_LHBEX	(R0)(R5),R10
	_LHBEX	(R0)(R6),R11
	_LHBEX	(R9)(R5),R12
	_LHBEX	(R9)(R6),R14
	RLDIMI	$32,R10,$0,R12
	RLDIMI	$32,R11,$0,R14
	CMPU	R12,R14
	BR	cmp0

	PCALIGN $16
cmp1:
	CMP	R9,$0
	BEQ	cmp0
	MOVBZ	(R5),R10
	MOVBZ	(R6),R11
	CMPU	R10,R11
cmp0:
	SETB_CR0(R6)
	ISEL	CR0EQ,R3,R6,R3
	RET
#endif

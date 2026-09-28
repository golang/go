// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// Register map
#define dstin	R0
#define src	R1
#define count	R2
#define dst	R3	// same as dstin, but gets modified in unaligned cases
#define srcend	R4	// points just past the last source byte
#define dstend	R5	// points just past the last destination byte
#define A_l	R6
#define A_h	R7
#define B_l	R8
#define B_h	R9
#define C_l	R10
#define C_h	R11
#define D_l	R12
#define D_h	R13
#define E_l	R14
#define E_h	R15
#define F_l	R16
#define F_h	R17
#define G_l	count
#define G_h	dst
#define H_l	src
#define H_h	srcend
#define tmp1	R14	// same as E_l
#define align_s	R7	// same as A_h; consumed before A is loaded
#define align_e	R8	// same as B_l; consumed before B is loaded

// Copies are split into 3 main cases: small copies of up to 32 bytes, medium
// copies of up to 128 bytes, and large copies. The overhead of the overlap
// check is negligible since it is only required for large copies.
//
// Large copies use a software pipelined loop processing 64 bytes per iteration.
// The destination pointer is 16-byte aligned to minimize unaligned accesses.
// The loop tail is handled by always copying 64 bytes from the end.

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-24
	CBZ	count, copy0

	// Small copies: 1..16 bytes
	CMP	$16, count
	BLE	copy16

	// Large copies
	CMP	$128, count
	BHI	copy_long
	CMP	$32, count
	BHI	copy32_128

	// Small copies: 17..32 bytes.
	LDP	(src), (A_l, A_h)
	ADD	src, count, srcend
	LDP	-16(srcend), (D_l, D_h)
	STP	(A_l, A_h), (dstin)
	ADD	dstin, count, dstend
	STP	(D_l, D_h), -16(dstend)
	RET

// Small copies: 1..16 bytes.
copy16:
	ADD	src, count, srcend
	ADD	dstin, count, dstend
	CMP	$8, count
	BLT	copy7
	MOVD	(src), A_l
	MOVD	-8(srcend), A_h
	MOVD	A_l, (dstin)
	MOVD	A_h, -8(dstend)
	RET

copy7:
	TBZ	$2, count, copy3
	MOVWU	(src), A_l
	MOVWU	-4(srcend), A_h
	MOVW	A_l, (dstin)
	MOVW	A_h, -4(dstend)
	RET

copy3:
	TBZ	$1, count, copy1
	MOVHU	(src), A_l
	MOVHU	-2(srcend), A_h
	MOVH	A_l, (dstin)
	MOVH	A_h, -2(dstend)
	RET

copy1:
	MOVBU	(src), A_l
	MOVB	A_l, (dstin)

copy0:
	RET

	// Medium copies: 33..128 bytes.
copy32_128:
	ADD	src, count, srcend
	ADD	dstin, count, dstend
	LDP	(src), (A_l, A_h)
	LDP	16(src), (B_l, B_h)
	LDP	-32(srcend), (C_l, C_h)
	LDP	-16(srcend), (D_l, D_h)
	CMP	$64, count
	BHI	copy128
	STP	(A_l, A_h), (dstin)
	STP	(B_l, B_h), 16(dstin)
	STP	(C_l, C_h), -32(dstend)
	STP	(D_l, D_h), -16(dstend)
	RET

	// Copy 65..128 bytes.
copy128:
	LDP	32(src), (E_l, E_h)
	LDP	48(src), (F_l, F_h)
	CMP	$96, count
	BLS	copy96
	LDP	-64(srcend), (G_l, G_h)
	LDP	-48(srcend), (H_l, H_h)
	STP	(G_l, G_h), -64(dstend)
	STP	(H_l, H_h), -48(dstend)

copy96:
	STP	(A_l, A_h), (dstin)
	STP	(B_l, B_h), 16(dstin)
	STP	(E_l, E_h), 32(dstin)
	STP	(F_l, F_h), 48(dstin)
	STP	(C_l, C_h), -32(dstend)
	STP	(D_l, D_h), -16(dstend)
	RET

	// Copy more than 128 bytes.
copy_long:
	ADD	src, count, srcend
	ADD	dstin, count, dstend
	MOVD	ZR, align_s
	MOVD	ZR, align_e

	CMP	$1024, count
	BLT	backward_check
	// feature detect to decide how to align
	MOVBU	runtime·arm64UseAlignedLoads(SB), A_l
	CBNZ	A_l, use_aligned_loads
	MOVD	dstin, align_s
	MOVD	dstend, align_e
	B	backward_check
use_aligned_loads:
	MOVD	src, align_s
	MOVD	srcend, align_e
	// align_s and align_e are used here for the realignment calculation. In
	// the use_aligned_loads case, align_s is the src pointer and align_e is
	// srcend pointer, which is used in the backward copy case.
	// When doing aligned stores, align_s is the dst pointer and align_e is
	// the dstend pointer.

backward_check:
	// Use backward copy if there is an overlap.
	SUB	src, dstin, tmp1
	CBZ	tmp1, copy0
	CMP	count, tmp1
	BCC	copy_long_backwards

	// Copy 16 bytes and then align src or dst to 16-byte alignment.
	LDP	(src), (D_l, D_h)
	AND	$15, align_s, tmp1	// Calculate the realignment offset
	SUB	tmp1, src, src
	SUB	tmp1, dstin, dst	// move dst back same amount as src
	ADD	tmp1, count, count
	LDP	16(src), (A_l, A_h)
	STP	(D_l, D_h), (dstin)
	LDP	32(src), (B_l, B_h)
	LDP	48(src), (C_l, C_h)
	LDP.W	64(src), (D_l, D_h)
	// 80 bytes have been loaded; if less than 80+64 bytes remain, copy from the end
	SUBS	$(80+64), count, count
	BLS	copy64_from_end

loop64:
	STP	(A_l, A_h), 16(dst)
	LDP	16(src), (A_l, A_h)
	STP	(B_l, B_h), 32(dst)
	LDP	32(src), (B_l, B_h)
	STP	(C_l, C_h), 48(dst)
	LDP	48(src), (C_l, C_h)
	STP.W	(D_l, D_h), 64(dst)
	LDP.W	64(src), (D_l, D_h)
	SUBS	$64, count, count
	BHI	loop64

	// Write the last iteration and copy 64 bytes from the end.
copy64_from_end:
	LDP	-64(srcend), (E_l, E_h)
	STP	(A_l, A_h), 16(dst)
	LDP	-48(srcend), (A_l, A_h)
	STP	(B_l, B_h), 32(dst)
	LDP	-32(srcend), (B_l, B_h)
	STP	(C_l, C_h), 48(dst)
	LDP	-16(srcend), (C_l, C_h)
	STP	(D_l, D_h), 64(dst)
	STP	(E_l, E_h), -64(dstend)
	STP	(A_l, A_h), -48(dstend)
	STP	(B_l, B_h), -32(dstend)
	STP	(C_l, C_h), -16(dstend)
	RET

	// Large backwards copy for overlapping copies.
	// Copy 16 bytes and then align srcend or dstend to 16-byte alignment.
copy_long_backwards:
	LDP	-16(srcend), (D_l, D_h)
	AND	$15, align_e, tmp1
	SUB	tmp1, srcend, srcend
	SUB	tmp1, count, count
	LDP	-16(srcend), (A_l, A_h)
	STP	(D_l, D_h), -16(dstend)
	LDP	-32(srcend), (B_l, B_h)
	LDP	-48(srcend), (C_l, C_h)
	LDP.W	-64(srcend), (D_l, D_h)
	SUB	tmp1, dstend, dstend
	SUBS	$128, count, count
	BLS	copy64_from_start

loop64_backwards:
	STP	(A_l, A_h), -16(dstend)
	LDP	-16(srcend), (A_l, A_h)
	STP	(B_l, B_h), -32(dstend)
	LDP	-32(srcend), (B_l, B_h)
	STP	(C_l, C_h), -48(dstend)
	LDP	-48(srcend), (C_l, C_h)
	STP.W	(D_l, D_h), -64(dstend)
	LDP.W	-64(srcend), (D_l, D_h)
	SUBS	$64, count, count
	BHI	loop64_backwards

	// Write the last iteration and copy 64 bytes from the start.
copy64_from_start:
	LDP	48(src), (G_l, G_h)
	STP	(A_l, A_h), -16(dstend)
	LDP	32(src), (A_l, A_h)
	STP	(B_l, B_h), -32(dstend)
	LDP	16(src), (B_l, B_h)
	STP	(C_l, C_h), -48(dstend)
	LDP	(src), (C_l, C_h)
	STP	(D_l, D_h), -64(dstend)
	STP	(G_l, G_h), 48(dstin)
	STP	(A_l, A_h), 32(dstin)
	STP	(B_l, B_h), 16(dstin)
	STP	(C_l, C_h), (dstin)
	RET

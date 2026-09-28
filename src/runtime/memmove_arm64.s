// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// Register map
#define dstin	R0
#define src	R1
#define count	R2
#define dst	R3
#define srcend	R4
#define dstend	R5
#define A_l	R6
#define A_h	R7
#define B_l	R8
#define B_h	R9
#define C_l	R10
#define align_s R11
#define align_e R12
#define tmp1	R14

#define A_q	F0
#define B_q	F1
#define C_q	F2
#define D_q	F3
#define E_q	F4
#define F_q	F5
#define G_q	F6
#define H_q	F7

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

	ADD	count, src, srcend
	ADD	count, dstin, dstend
	// Small copies: 1..16 bytes
	CMP	$16, count
	BLE	copy16
	// Large copies:
	CMP	$128, count
	BHI	copy_long
	CMP	$32, count
	BHI	copy32_128

	// Small copies: 17..32 bytes.
	LDP	(src), (A_l, A_h)
	LDP	-16(srcend), (B_l, B_h)
	STP	(A_l, A_h), (dstin)
	STP	(B_l, B_h), -16(dstend)
	RET

	// Small copies: 1..16 bytes.
copy16:
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
	MOVWU	-4(srcend), B_l
	MOVW	A_l, (dstin)
	MOVW	B_l, -4(dstend)
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
	FLDPQ	(src), (A_q, B_q)
	FLDPQ	-32(srcend), (C_q, D_q)
	CMP	$64, count
	BHI	copy128
	FSTPQ	(A_q, B_q), (dstin)
	FSTPQ	(C_q, D_q), -32(dstend)
	RET

	// Copy 65..128 bytes.
copy128:
	FLDPQ	32(src), (E_q, F_q)
	CMP	$96, count
	BLS	copy96
	FLDPQ	-64(srcend), (G_q, H_q)
	FSTPQ	(G_q, H_q), -64(dstend)

copy96:
	FSTPQ	(A_q, B_q), (dstin)
	FSTPQ	(E_q, F_q), 32(dstin)
	FSTPQ	(C_q, D_q), -32(dstend)
	RET

	// Copy more than 128 bytes.
copy_long:
	// Neoverse cores (arm64UseAlignedLoads) copy faster with aligned
	// loads at every size. Other cores align the stores, and only for
	// copies of 1024 bytes or more.
	MOVBU	runtime·arm64UseAlignedLoads(SB), tmp1
	CBNZ	tmp1, use_aligned_loads
	MOVD	ZR, align_s
	MOVD	ZR, align_e
	CMP	$1024, count
	BLT	backward_check
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
	// Use backwards copy if there is an overlap.
	SUB	src, dstin, tmp1
	CMP	count, tmp1
	BLO	copy_long_backwards

	// Copy 16 bytes and then align src or dst to 16-byte alignment.
	FMOVQ	(src), D_q
	AND	$15, align_s, tmp1
	SUB	tmp1, src, src
	SUB	tmp1, dstin, dst
	ADD	tmp1, count, count  // count is now 16 too large.
	FLDPQ	16(src), (A_q, B_q)
	FMOVQ	D_q, (dstin)
	FLDPQ	48(src), (C_q, D_q)
	// 80 bytes have been loaded; if less than 80+64 bytes remain, copy from the end.
	SUBS	$(80+64), count, count
	BLS	copy64_from_end

	// Pin loop alignment so unrelated edits can't shift it.
	PCALIGN	$32
loop64:
	FSTPQ	(A_q, B_q), 16(dst)
	FLDPQ	80(src), (A_q, B_q)
	FSTPQ	(C_q, D_q), 48(dst)
	FLDPQ	112(src), (C_q, D_q)
	ADD	$64, src, src
	ADD	$64, dst, dst
	SUBS	$64, count, count
	BHI	loop64

	// Write the last iteration and copy 64 bytes from the end.
copy64_from_end:
	FLDPQ	-64(srcend), (E_q, F_q)
	FSTPQ	(A_q, B_q), 16(dst)
	FLDPQ	-32(srcend), (A_q, B_q)
	FSTPQ	(C_q, D_q), 48(dst)
	FSTPQ	(E_q, F_q), -64(dstend)
	FSTPQ	(A_q, B_q), -32(dstend)
	RET

	// Large backwards copy for overlapping copies.
	// Copy 16 bytes and then align srcend  or dstend to 16-byte alignment.
copy_long_backwards:
	CBZ	tmp1, copy0
	FMOVQ	-16(srcend), D_q
	AND	$15, align_e, tmp1
	SUB	tmp1, srcend, srcend
	SUB	tmp1, count, count
	FLDPQ	-32(srcend), (A_q, B_q)
	FMOVQ	D_q, -16(dstend)
	FLDPQ	-64(srcend), (C_q, D_q)
	SUB	tmp1, dstend, dstend
	SUBS	$128, count, count
	BLS	copy64_from_start

	PCALIGN	$32
loop64_backwards:
	FMOVQ	B_q, -16(dstend)
	FMOVQ	A_q, -32(dstend)
	FLDPQ	-96(srcend), (A_q, B_q)
	FMOVQ	D_q, -48(dstend)
	FMOVQ.W	C_q, -64(dstend)
	FLDPQ	-128(srcend), (C_q, D_q)
	SUB	$64, srcend, srcend
	SUBS	$64, count, count
	BHI	loop64_backwards

	// Write the last iteration and copy 64 bytes from the start
copy64_from_start:
	FLDPQ	32(src), (E_q, F_q)
	FSTPQ	(A_q, B_q), -32(dstend)
	FLDPQ	(src), (A_q, B_q)
	FSTPQ	(C_q, D_q), -64(dstend)
	FSTPQ	(E_q, F_q), 32(dstin)
	FSTPQ	(A_q, B_q), (dstin)
	RET

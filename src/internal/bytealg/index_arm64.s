// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// func Index(a, b []byte) int
// input:
//   R0: a ptr (haystack)
//   R1: a len (haystack)
//   R2: a cap (haystack) (unused)
//   R3: b ptr (needle)
//   R4: b len (needle) (2 <= len <= 32)
//   R5: b cap (needle) (unused)
// return:
//   R0: result
TEXT ·Index<ABIInternal>(SB),NOSPLIT,$0-56
	MOVD	R3, R2
	MOVD	R4, R3
	B	·IndexString<ABIInternal>(SB)

// func IndexString(a, b string) int
// input:
//   R0: a ptr (haystack)
//   R1: a len (haystack)
//   R2: b ptr (needle)
//   R3: b len (needle) (2 <= len <= 32)
// return:
//   R0: result
TEXT ·IndexString<ABIInternal>(SB),NOSPLIT,$0-40
	SUB	R3, R1, R4
	// R4 = len(a) - len(b) is the last valid start offset of a match.
	// If there are at least 16 valid start offsets, use the SIMD
	// search; otherwise fall through to the scalar search below.
	CMP	$15, R4
	BGE	simd

	// main idea is to load 'sep' into separate register(s)
	// to avoid repeatedly re-load it again and again
	// for sebsequent substring comparisons
	// R4 contains the start of last substring for comparison
	ADD	R0, R4, R4
	ADD	$1, R0, R8

	CMP	$8, R3
	BHI	greater_8
	TBZ	$3, R3, len_2_7
len_8:
	// R5 contains 8-byte of sep
	MOVD	(R2), R5
loop_8:
	// R6 contains substring for comparison
	CMP	R4, R0
	BHI	not_found
	MOVD.P	1(R0), R6
	CMP	R5, R6
	BNE	loop_8
	B	found
len_2_7:
	TBZ	$2, R3, len_2_3
	TBZ	$1, R3, len_4_5
	TBZ	$0, R3, len_6
len_7:
	// R5 and R6 contain 7-byte of sep
	MOVWU	(R2), R5
	// 1-byte overlap with R5
	MOVWU	3(R2), R6
loop_7:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	loop_7
	MOVWU	2(R0), R3
	CMP	R6, R3
	BNE	loop_7
	B	found
len_6:
	// R5 and R6 contain 6-byte of sep
	MOVWU	(R2), R5
	MOVHU	4(R2), R6
loop_6:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	loop_6
	MOVHU	3(R0), R3
	CMP	R6, R3
	BNE	loop_6
	B	found
len_4_5:
	TBZ	$0, R3, len_4
len_5:
	// R5 and R7 contain 5-byte of sep
	MOVWU	(R2), R5
	MOVBU	4(R2), R7
loop_5:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R3
	CMP	R5, R3
	BNE	loop_5
	MOVBU	3(R0), R3
	CMP	R7, R3
	BNE	loop_5
	B	found
len_4:
	// R5 contains 4-byte of sep
	MOVWU	(R2), R5
loop_4:
	CMP	R4, R0
	BHI	not_found
	MOVWU.P	1(R0), R6
	CMP	R5, R6
	BNE	loop_4
	B	found
len_2_3:
	TBZ	$0, R3, len_2
len_3:
	// R6 and R7 contain 3-byte of sep
	MOVHU	(R2), R6
	MOVBU	2(R2), R7
loop_3:
	CMP	R4, R0
	BHI	not_found
	MOVHU.P	1(R0), R3
	CMP	R6, R3
	BNE	loop_3
	MOVBU	1(R0), R3
	CMP	R7, R3
	BNE	loop_3
	B	found
len_2:
	// R5 contains 2-byte of sep
	MOVHU	(R2), R5
loop_2:
	CMP	R4, R0
	BHI	not_found
	MOVHU.P	1(R0), R6
	CMP	R5, R6
	BNE	loop_2
found:
	SUB	R8, R0, R0
	RET
not_found:
	MOVD	$-1, R0
	RET
greater_8:
	SUB	$9, R3, R11	// len(sep) - 9, offset of R0 for last 8 bytes
	CMP	$16, R3
	BHI	greater_16
len_9_16:
	MOVD.P	8(R2), R5	// R5 contains the first 8-byte of sep
	SUB	$16, R3, R7	// len(sep) - 16, offset of R2 for last 8 bytes
	MOVD	(R2)(R7), R6	// R6 contains the last 8-byte of sep
loop_9_16:
	// search the first 8 bytes first
	CMP	R4, R0
	BHI	not_found
	MOVD.P	1(R0), R7
	CMP	R5, R7
	BNE	loop_9_16
	MOVD	(R0)(R11), R7
	CMP	R6, R7		// compare the last 8 bytes
	BNE	loop_9_16
	B	found
greater_16:
	CMP	$24, R3
	BHI	len_25_32
len_17_24:
	LDP.P	16(R2), (R5, R6)	// R5 and R6 contain the first 16-byte of sep
	SUB	$24, R3, R10		// len(sep) - 24
	MOVD	(R2)(R10), R7		// R7 contains the last 8-byte of sep
loop_17_24:
	// search the first 16 bytes first
	CMP	R4, R0
	BHI	not_found
	MOVD.P	1(R0), R10
	CMP	R5, R10
	BNE	loop_17_24
	MOVD	7(R0), R10
	CMP	R6, R10
	BNE	loop_17_24
	MOVD	(R0)(R11), R10
	CMP	R7, R10		// compare the last 8 bytes
	BNE	loop_17_24
	B	found
len_25_32:
	LDP.P	16(R2), (R5, R6)
	MOVD.P	8(R2), R7	// R5, R6 and R7 contain the first 24-byte of sep
	SUB	$32, R3, R12	// len(sep) - 32
	MOVD	(R2)(R12), R10	// R10 contains the last 8-byte of sep
loop_25_32:
	// search the first 24 bytes first
	CMP	R4, R0
	BHI	not_found
	MOVD.P	1(R0), R12
	CMP	R5, R12
	BNE	loop_25_32
	MOVD	7(R0), R12
	CMP	R6, R12
	BNE	loop_25_32
	MOVD	15(R0), R12
	CMP	R7, R12
	BNE	loop_25_32
	MOVD	(R0)(R11), R12
	CMP	R10, R12	// compare the last 8 bytes
	BNE	loop_25_32
	B	found

	// SIMD (NEON) search, used when there are at least 16 valid start
	// offsets, i.e. len(a)-len(b) >= 15.
	//
	// Instead of checking one start offset per iteration like the
	// scalar loops above, each iteration checks two "anchor" bytes of
	// the needle -- the first and the last -- at 16 consecutive start
	// offsets at once, and runs a full comparison only at offsets
	// where both anchors match.
	//
	// For a block of start offsets [i, i+15] (n = len(b)):
	//	VLD1  a[i:i+16]     -> VCMEQ against b[0]    (broadcast in V0)
	//	VLD1  a[i+n-1:i+n+15] -> VCMEQ against b[n-1] (broadcast in V1)
	//	VAND: lane L is 0xff iff a[i+L]==b[0] && a[i+L+n-1]==b[n-1]
	// VSHRN $4 on the .H8 view of the AND result narrows each pair of
	// 0x00/0xff byte lanes into one byte holding two 0x0/0xf nibbles,
	// producing a 64-bit mask in which bits [4L,4L+4) correspond to
	// lane L (little-endian). The mask is scanned with RBIT+CLZ: a
	// matching lane's lowest set bit is at bit 4L exactly, so the bit
	// index >> 2 recovers the lane, hence the candidate start offset.
	//
	// The final block starts at &a[last-15] (last = len(a)-len(b)) so
	// its 16 lanes are exactly the last 16 valid start offsets: it may
	// overlap already-checked offsets, but never checks invalid ones,
	// so no lane masking is needed anywhere. A candidate in the overlap
	// that failed verification in the previous block is verified again.
	// TODO: mask the overlapping lanes out of the final block's mask,
	// if that ever shows up for long needles.
	//
	// When a block has anchor matches, needles of length >= 4 are
	// first refined with two more anchors (b[1] and b[len(b)-2]),
	// still 16 lanes at a time, so blocks of pure anchor false
	// positives are usually rejected without per-candidate work.
	// Remaining candidates are verified with loads of b-sized chunks
	// anchored to its start and end, covering all of b:
	//	{2,3}:    one 2-byte chunk (the anchors cover the rest)
	//	{4..8}:   two 4-byte chunks; one 4-byte chunk at max(len(b)-6, 0)
	//	          after the refine pass, since that has already matched
	//	          b[0:2] and b[len(b)-2:]
	//	{9..16}:  two 8-byte chunks
	//	{17..32}: four 8-byte chunks
	// For 9 or more bytes the refined anchors do not reduce the number
	// of 8-byte loads needed to cover the rest of b.
	//
	// All loads stay within a[0:len(a)]:
	//	anchor A: P <= &a[last-15], reads through &a[last] = &a[len(a)-n], n >= 2
	//	anchor B: P+n-1 reads through &a[last-15+n-1+15] = &a[len(a)-1]
	//	refine anchors at P+1 and P+n-2 read strictly inside the above
	//	  (n >= 4 whenever the refinement runs)
	// Candidate verify chunks at offset j <= last use a chunk width w
	// with len(b) >= w and offsets in [0, len(b)-w], so every load is
	// within a[j:j+len(b)] and j+len(b) <= len(a); the same offsets are
	// used to load the chunks of b itself.
	//
	// Register plan:
	// R0 = &a[0] (preserved to compute the result)
	// R2 = &b[0], R3 = len(b) (setup only)
	// R5 = &a[last-15], start of the final block
	// R6 = len(b)-1, offset of the anchor B load
	// R8 = P, current block pointer
	// R9, R11 (and R1, R7 for {17..32}) = verify chunks of b
	//   ({2,3} only uses R9, its whole verify chunk fits in one check)
	// R10 (and R17 for {17..32}) = verify chunk offsets
	// R12-R16 = per-iteration scratch
	// V0, V1 = broadcast primary anchors b[0], b[len(b)-1]
	// V6, V7 = broadcast refinement anchors b[1], b[len(b)-2]
	// V2 = anchor match mask (byte lanes); V3, V4, V5 = scratch
simd:
	MOVBU	(R2), R5	// b[0], anchor byte A
	SUB	$1, R3, R6
	MOVBU	(R2)(R6), R7	// b[len(b)-1], anchor byte B
	VDUP	R5, V0.B16
	VDUP	R7, V1.B16
	ADD	R0, R4, R5
	SUB	$15, R5, R5	// R5 = &a[last-15]
	MOVD	R0, R8
	CMP	$8, R3
	BHI	simd_9plus
	CMP	$3, R3
	BHI	simd_4_8

	// 2 <= len(b) <= 3: the two anchors (b[0] and b[len(b)-1]) already
	// cover a 2-byte needle in full. For a 3-byte needle they leave
	// only the middle byte (b[1]) unverified; the single 2-byte chunk
	// check below covers it too (candidate[0:2] vs b[0:2]): its first
	// byte is redundant with anchor A, its second is the middle byte.
	MOVHU	(R2), R9
simd_loop_2_3:
	VLD1	(R8), [V2.B16]
	ADD	R6, R8, R12
	VLD1	(R12), [V3.B16]
	VCMEQ	V0.B16, V2.B16, V2.B16
	VCMEQ	V1.B16, V3.B16, V3.B16
	VAND	V3.B16, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V2.B8
	VMOV	V2.D[0], R13
	CBNZ	R13, simd_cand_2_3
simd_next_2_3:
	CMP	R5, R8
	BEQ	not_found
	ADD	$16, R8
	CMP	R5, R8
	CSEL	LS, R8, R5, R8	// P = min(P+16, &a[last-15])
	B	simd_loop_2_3
simd_cand_2_3:
	RBIT	R13, R14
	CLZ	R14, R14	// R14 = 4 * (matching lane)
	ADD	R14>>2, R8, R15	// R15 = candidate = &a[j]
	MOVHU	(R15), R16
	CMP	R9, R16
	BNE	simd_cand_next_2_3
	SUB	R0, R15, R0
	RET
simd_cand_next_2_3:
	MOVD	$0xF, R16
	LSL	R14, R16, R16
	BIC	R16, R13, R13	// clear this lane's nibble
	CBNZ	R13, simd_cand_2_3
	B	simd_next_2_3

	// 4 <= len(b) <= 8: verify with two overlapping 4-byte chunks.
simd_4_8:
	MOVBU	1(R2), R7	// b[1], refinement anchor
	VDUP	R7, V6.B16
	SUB	$1, R6, R7
	MOVBU	(R2)(R7), R7	// b[len(b)-2], refinement anchor
	VDUP	R7, V7.B16
	// needle portions for lone candidate check
	SUB	$4, R3, R10
	MOVWU	(R2), R9
	MOVWU	(R2)(R10), R11
	// needle portion for multiple candidate check: the 4 bytes at
	// max(len(b)-6, 0) cover whatever's left unverified once all four
	// anchor bytes (b[0], b[1], b[len(b)-2], b[len(b)-1]) are confirmed
	// by the refine pass.
	SUB	$6, R3, R1
	CMP	$0, R1
	CSEL	GE, R1, ZR, R1
	MOVWU	(R2)(R1), R17
simd_loop_4_8:
	VLD1	(R8), [V2.B16]
	ADD	R6, R8, R12
	VLD1	(R12), [V3.B16]
	VCMEQ	V0.B16, V2.B16, V2.B16
	VCMEQ	V1.B16, V3.B16, V3.B16
	VAND	V3.B16, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V5.B8
	VMOV	V5.D[0], R13
	CBNZ	R13, simd_maybe_refine_4_8
simd_next_4_8:
	CMP	R5, R8
	BEQ	not_found
	ADD	$16, R8
	CMP	R5, R8
	CSEL	LS, R8, R5, R8	// P = min(P+16, &a[last-15])
	B	simd_loop_4_8
simd_maybe_refine_4_8:
	// Some start offset in this block matched both anchors. A lone
	// candidate (the common case in ordinary text, e.g. the actual
	// match) is cheaper to verify directly -- it only has the two
	// primary anchors confirmed, so it needs the full two-chunk check
	// (simd_cand_4_8_lone). For multiple candidates, prune with two
	// more anchors (b[1] and b[len(b)-2]) while still 16 lanes at a
	// time; blocks of pure anchor false positives are usually
	// rejected wholesale here, and candidates that survive have all
	// four anchor bytes confirmed, enabling the cheaper single-check
	// verify (simd_cand_4_8_refined). Matching lanes hold 0xF, so
	// isolating one bit per lane and clearing the lowest tells lone
	// from multiple.
	AND	$0x1111111111111111, R13, R14
	SUB	$1, R14, R16
	TST	R16, R14
	BEQ	simd_cand_4_8_lone
	SUB	$1, R12, R12	// &a[i+len(b)-2]
	VLD1	(R12), [V3.B16]
	ADD	$1, R8, R12	// &a[i+1]
	VLD1	(R12), [V4.B16]
	VCMEQ	V7.B16, V3.B16, V3.B16
	VCMEQ	V6.B16, V4.B16, V4.B16
	VAND	V4.B16, V3.B16, V3.B16
	VAND	V3.B16, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V5.B8
	VMOV	V5.D[0], R13
	CBZ	R13, simd_next_4_8
simd_cand_4_8_refined:
	RBIT	R13, R14
	CLZ	R14, R14	// R14 = 4 * (matching lane)
	ADD	R14>>2, R8, R15	// R15 = candidate = &a[j]
	MOVWU	(R15)(R1), R16
	CMP	R17, R16
	BNE	simd_cand_next_4_8_refined
	SUB	R0, R15, R0
	RET
simd_cand_next_4_8_refined:
	MOVD	$0xF, R16
	LSL	R14, R16, R16
	BIC	R16, R13, R13	// clear this lane's nibble
	CBNZ	R13, simd_cand_4_8_refined
	B	simd_next_4_8
simd_cand_4_8_lone:
	RBIT	R13, R14
	CLZ	R14, R14	// R14 = 4 * (matching lane)
	ADD	R14>>2, R8, R15	// R15 = candidate = &a[j]
	MOVWU	(R15), R16
	CMP	R9, R16
	BNE	simd_next_4_8	// it was the block's only candidate
	MOVWU	(R15)(R10), R16
	CMP	R11, R16
	BNE	simd_next_4_8
	SUB	R0, R15, R0
	RET

simd_9plus:
	MOVBU	1(R2), R7	// b[1], refinement anchor
	VDUP	R7, V6.B16
	SUB	$1, R6, R7
	MOVBU	(R2)(R7), R7	// b[len(b)-2], refinement anchor
	VDUP	R7, V7.B16
	CMP	$16, R3
	BHI	simd_17_32

	// 9 <= len(b) <= 16: verify with two overlapping 8-byte chunks.
	SUB	$8, R3, R10
	MOVD	(R2), R9
	MOVD	(R2)(R10), R11
simd_loop_9_16:
	VLD1	(R8), [V2.B16]
	ADD	R6, R8, R12
	VLD1	(R12), [V3.B16]
	VCMEQ	V0.B16, V2.B16, V2.B16
	VCMEQ	V1.B16, V3.B16, V3.B16
	VAND	V3.B16, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V5.B8
	VMOV	V5.D[0], R13
	CBNZ	R13, simd_maybe_refine_9_16
simd_next_9_16:
	CMP	R5, R8
	BEQ	not_found
	ADD	$16, R8
	CMP	R5, R8
	CSEL	LS, R8, R5, R8	// P = min(P+16, &a[last-15])
	B	simd_loop_9_16
simd_maybe_refine_9_16:
	// See the comment in simd_maybe_refine_4_8.
	AND	$0x1111111111111111, R13, R14
	SUB	$1, R14, R16
	TST	R16, R14
	BEQ	simd_cand_9_16
	SUB	$1, R12, R12	// &a[i+len(b)-2]
	VLD1	(R12), [V3.B16]
	ADD	$1, R8, R12	// &a[i+1]
	VLD1	(R12), [V4.B16]
	VCMEQ	V7.B16, V3.B16, V3.B16
	VCMEQ	V6.B16, V4.B16, V4.B16
	VAND	V4.B16, V3.B16, V3.B16
	VAND	V3.B16, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V5.B8
	VMOV	V5.D[0], R13
	CBZ	R13, simd_next_9_16
simd_cand_9_16:
	RBIT	R13, R14
	CLZ	R14, R14	// R14 = 4 * (matching lane)
	ADD	R14>>2, R8, R15	// R15 = candidate = &a[j]
	MOVD	(R15), R16
	CMP	R9, R16
	BNE	simd_cand_next_9_16
	MOVD	(R15)(R10), R16
	CMP	R11, R16
	BNE	simd_cand_next_9_16
	SUB	R0, R15, R0
	RET
simd_cand_next_9_16:
	MOVD	$0xF, R16
	LSL	R14, R16, R16
	BIC	R16, R13, R13	// clear this lane's nibble
	CBNZ	R13, simd_cand_9_16
	B	simd_next_9_16

	// 17 <= len(b) <= 32: verify with four overlapping 8-byte chunks
	// covering b[0:16] and b[len(b)-16:len(b)], with an early exit on
	// the first mismatching chunk.
simd_17_32:
	SUB	$16, R3, R10
	SUB	$8, R3, R17
	MOVD	(R2), R9
	MOVD	8(R2), R11
	MOVD	(R2)(R10), R1
	MOVD	(R2)(R17), R7
simd_loop_17_32:
	VLD1	(R8), [V2.B16]
	ADD	R6, R8, R12
	VLD1	(R12), [V3.B16]
	VCMEQ	V0.B16, V2.B16, V2.B16
	VCMEQ	V1.B16, V3.B16, V3.B16
	VAND	V3.B16, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V5.B8
	VMOV	V5.D[0], R13
	CBNZ	R13, simd_maybe_refine_17_32
simd_next_17_32:
	CMP	R5, R8
	BEQ	not_found
	ADD	$16, R8
	CMP	R5, R8
	CSEL	LS, R8, R5, R8	// P = min(P+16, &a[last-15])
	B	simd_loop_17_32
simd_maybe_refine_17_32:
	// See the comment in simd_maybe_refine_4_8.
	AND	$0x1111111111111111, R13, R14
	SUB	$1, R14, R16
	TST	R16, R14
	BEQ	simd_cand_17_32
	SUB	$1, R12, R12	// &a[i+len(b)-2]
	VLD1	(R12), [V3.B16]
	ADD	$1, R8, R12	// &a[i+1]
	VLD1	(R12), [V4.B16]
	VCMEQ	V7.B16, V3.B16, V3.B16
	VCMEQ	V6.B16, V4.B16, V4.B16
	VAND	V4.B16, V3.B16, V3.B16
	VAND	V3.B16, V2.B16, V2.B16
	VSHRN	$4, V2.H8, V5.B8
	VMOV	V5.D[0], R13
	CBZ	R13, simd_next_17_32
simd_cand_17_32:
	RBIT	R13, R14
	CLZ	R14, R14	// R14 = 4 * (matching lane)
	ADD	R14>>2, R8, R15	// R15 = candidate = &a[j]
	MOVD	(R15), R16
	CMP	R9, R16
	BNE	simd_cand_next_17_32
	MOVD	8(R15), R16
	CMP	R11, R16
	BNE	simd_cand_next_17_32
	MOVD	(R15)(R10), R16
	CMP	R1, R16
	BNE	simd_cand_next_17_32
	MOVD	(R15)(R17), R16
	CMP	R7, R16
	BNE	simd_cand_next_17_32
	SUB	R0, R15, R0
	RET
simd_cand_next_17_32:
	MOVD	$0xF, R16
	LSL	R14, R16, R16
	BIC	R16, R13, R13	// clear this lane's nibble
	CBNZ	R13, simd_cand_17_32
	B	simd_next_17_32

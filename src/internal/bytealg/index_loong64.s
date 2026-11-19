// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Index<ABIInternal>(SB),NOSPLIT,$0-56
	MOVV	R7, R6		// R6 = separator pointer
	MOVV	R8, R7		// R7 = separator length
	JMP	indexbody<>(SB)

TEXT ·IndexString<ABIInternal>(SB),NOSPLIT,$0-40
	JMP	indexbody<>(SB)

// input:
//   R4 = string
//   R5 = length
//   R6 = separator pointer
//   R7 = separator length (2 <= len <= 64)
TEXT indexbody<>(SB),NOSPLIT,$0
	// main idea is to load 'sep' into separate register(s)
	// to avoid repeatedly re-load it again and again
	// for sebsequent substring comparisons
	SUBV	R7, R5, R8
	ADDV	R4, R8		// R8 contains the start of last substring for comparison
	ADDV	$1, R4, R9	// store base for later

	// For small haystacks the scalar loops below are faster;
	// skip the SIMD dispatch entirely. The SIMD setup cost
	// (feature check, two broadcasts) only pays off for
	// hlen >= 64.
	MOVV	$64, R10
	BLT	R5, R10, scalar_dispatch
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R10
	BNE	R10, lasx_index
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R10
	BNE	R10, lsx_index
scalar_dispatch:
	MOVV	$8, R5
	BGE	R7, R5, len_gt_or_eq_8
len_2_7:
	AND	$0x4, R7, R5
	BNE	R5, len_4_7

len_2_3:
	AND	$0x1, R7, R5
	BNE	R5, len_3

len_2:
	MOVHU	(R6), R10
loop_2:
	BLT	R8, R4, not_found
	MOVHU	(R4), R11
	ADDV	$1, R4
	BNE	R10, R11, loop_2
	JMP	found

len_3:
	MOVHU	(R6), R10
	MOVBU	2(R6), R11
loop_3:
	BLT	R8, R4, not_found
	MOVHU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_3
	MOVBU	1(R4), R13
	BNE	R11, R13, loop_3
	JMP	found

len_4_7:
	AND	$0x2, R7, R5
	BNE	R5, len_6_7
	AND	$0x1, R7, R5
	BNE	R5, len_5
len_4:
	MOVWU	(R6), R10
loop_4:
	BLT	R8, R4, not_found
	MOVWU	(R4), R11
	ADDV	$1, R4
	BNE	R10, R11, loop_4
	JMP	found

len_5:
	MOVWU	(R6), R10
	MOVBU	4(R6), R11
loop_5:
	BLT	R8, R4, not_found
	MOVWU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_5
	MOVBU	3(R4), R13
	BNE	R11, R13, loop_5
	JMP	found

len_6_7:
	AND	$0x1, R7, R5
	BNE	R5, len_7
len_6:
	MOVWU	(R6), R10
	MOVHU	4(R6), R11
loop_6:
	BLT	R8, R4, not_found
	MOVWU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_6
	MOVHU	3(R4), R13
	BNE	R11, R13, loop_6
	JMP	found

len_7:
	MOVWU	(R6), R10
	MOVWU	3(R6), R11
loop_7:
	BLT	R8, R4, not_found
	MOVWU	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_7
	MOVWU	2(R4), R13
	BNE	R11, R13, loop_7
	JMP	found

len_gt_or_eq_8:
	BEQ	R5, R7, len_8
	MOVV	$17, R5
	BGE	R7, R5, len_gt_or_eq_17
	JMP	len_9_16
len_8:
	MOVV	(R6), R10
loop_8:
	BLT	R8, R4, not_found
	MOVV	(R4), R11
	ADDV	$1, R4
	BNE	R10, R11, loop_8
	JMP	found

len_9_16:
	MOVV	(R6), R10
	SUBV	$8, R7
	MOVV	(R6)(R7), R11
	SUBV	$1, R7
loop_9_16:
	BLT	R8, R4, not_found
	MOVV	(R4), R12
	ADDV	$1, R4
	BNE	R10, R12, loop_9_16
	MOVV	(R4)(R7), R13
	BNE	R11, R13, loop_9_16
	JMP	found

len_gt_or_eq_17:
	MOVV	$25, R5
	BGE	R7, R5, len_gt_or_eq_25
len_17_24:
	MOVV	0(R6), R10
	MOVV	8(R6), R11
	SUBV	$8, R7
	MOVV	(R6)(R7), R12
	SUBV	$1, R7
loop_17_24:
	BLT	R8, R4, not_found
	MOVV	(R4), R13
	ADDV	$1, R4
	BNE	R10, R13, loop_17_24
	MOVV	7(R4), R14
	BNE	R11, R14, loop_17_24
	MOVV	(R4)(R7), R15
	BNE	R12, R15, loop_17_24
	JMP	found

len_gt_or_eq_25:
	MOVV	$33, R5
	BGE	R7, R5, len_gt_or_eq_33
	MOVBU   internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R10
	BNE	R10, lsx_len_25_32
len_25_32:
	MOVV	0(R6), R10
	MOVV	8(R6), R11
	MOVV	16(R6), R12
	SUBV	$8, R7
	MOVV	(R6)(R7), R13
	SUBV	$1, R7
loop_25_32:
	BLT	R8, R4, not_found
	MOVV	(R4), R14
	ADDV	$1, R4
	BNE	R10, R14, loop_25_32
	MOVV	7(R4), R15
	BNE	R11, R15, loop_25_32
	MOVV	15(R4), R16
	BNE	R12, R16, loop_25_32
	MOVV	(R4)(R7), R17
	BNE	R13, R17, loop_25_32
	JMP	found

	// On loong64, LSX is included if LASX is supported.
lasx_len_25_32:
lsx_len_25_32:
	VMOVQ	0(R6), V0
	SUBV	$16, R7
	VMOVQ	(R6)(R7), V1
	SUBV	$1, R7
lsx_loop_25_32:
	BLT	R8, R4, not_found
	VMOVQ	(R4), V2
	ADDV	$1, R4
	VSEQV	V0, V2, V2
	VSETANYEQV	V2, FCC0
	BFPT	FCC0, lsx_loop_25_32

	VMOVQ	(R4)(R7), V3
	VSEQV	V1, V3, V3
	VSETANYEQV	V3, FCC1
	BFPT	FCC1, lsx_loop_25_32
	JMP	found

len_gt_or_eq_33:
	// seplen > 32 has no scalar loop. It only reaches
	// indexbody when SIMD is available (MaxLen == 32
	// without SIMD, see index_loong64.go), so dispatch
	// directly to the SIMD paths. This is reachable for
	// hlen < 64 via scalar_dispatch.
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R10
	BNE	R10, lasx_index
	JMP	lsx_index

found:
	SUBV	R9, R4
	RET

not_found:
	MOVV	$-1, R4
	RET

// --------------------------------------------------------------------
// SIMD first+last byte filter paths.
//
// Broadcast sep[0] and sep[seplen-1] into vector registers. Each scan
// step compares a 16/32-byte block at R4 against sep[0] AND the block
// at R4+seplen-1 against sep[seplen-1]; a candidate requires both to
// match, which makes false candidates rare even when sep[0] is common.
// Candidates in a block are consumed one by one from the bitmask
// (lowest set bit via CTZV, cleared with mask &= mask-1) and verified
// with scalar compares (word-wise for seplen >= 8, byte-wise below).
//
// Registers:
//   R4  = current scan pointer
//   R8  = last valid start position (base + hlen - slen)
//   R9  = base + 1 (for result computation)
//   R6  = separator pointer, R7 = separator length
//   R10 = R7 - 1 (offset of last sep byte)
//   R12 = last scan pointer where both vector loads stay in bounds
//         (the load at R4+seplen-1 reads up to R4+seplen-1+15/31,
//         so R12 = R8 - 15 for LSX, R8 - 31 for LASX)
//   R13 = candidate position, R14 = candidate bitmask
//   R15-R18 = scratch, R19 = candidate budget per block

lsx_index:
	MOVBU	(R6), R10
	VMOVQ	R10, V0.B16		// broadcast sep[0]
	ADDV	$-1, R7, R10
	MOVBU	(R6)(R10), R11
	VMOVQ	R11, V1.B16		// broadcast sep[seplen-1]
	ADDV	$-15, R8, R12		// both loads in bounds while R4 <= R12

lsx_scan:
	BLT	R12, R4, lsx_tail
	VMOVQ	(R4), V2
	VSEQB	V2, V0, V2		// sep[0] matches
	VMOVQ	(R4)(R10), V3
	VSEQB	V3, V1, V3		// sep[last] matches
	VANDV	V3, V2, V2		// candidate = both matched
	VMSKLTZB	V2, V4
	VMOVQ	V4.V[0], R14		// 16-bit candidate mask
	BEQ	R14, lsx_advance
	MOVV	$4, R19			// candidate budget per block

lsx_cand:
	CTZV	R14, R15		// lowest candidate byte offset
	ADDV	R4, R15, R13
	BLT	R8, R13, not_found	// later candidates are also beyond R8
	// verify full separator at R13
	MOVV	$8, R15
	BLT	R7, R15, lsx_verify_bytes
	// wide verify: 8-byte words at 0,8,16,... plus last word at R7-8
	MOVV	(R6), R16
	MOVV	(R13), R17
	BNE	R16, R17, lsx_cand_next
	ADDV	$-8, R7, R18		// offset of last word
	MOVV	$8, R15
lsx_verify_loop:
	BGE	R15, R18, lsx_verify_last
	MOVV	(R6)(R15), R16
	MOVV	(R13)(R15), R17
	BNE	R16, R17, lsx_cand_next
	ADDV	$8, R15
	JMP	lsx_verify_loop
lsx_verify_last:
	MOVV	(R6)(R18), R16
	MOVV	(R13)(R18), R17
	BNE	R16, R17, lsx_cand_next
	JMP	lsx_hit
lsx_verify_bytes:			// seplen < 8: byte-wise verify
	MOVV	R0, R15
lsx_verify_bloop:
	MOVBU	(R6)(R15), R16
	MOVBU	(R13)(R15), R17
	BNE	R16, R17, lsx_cand_next
	ADDV	$1, R15
	BNE	R15, R7, lsx_verify_bloop
lsx_hit:
	MOVV	R13, R4
	ADDV	$1, R4			// found expects R4 = pos + 1
	JMP	found
lsx_cand_next:
	ADDV	$-1, R14, R15
	AND	R15, R14		// clear lowest candidate bit
	BEQ	R14, lsx_advance
	// Too many candidates in one block (e.g. periodic
	// haystacks): per-candidate verify costs more than the
	// scalar loops, so finish the string scalar. Resume at
	// Resume at R13+1: positions before R13 in this block either
	// failed the first+last byte filter or were already verified
	// and rejected, so none of them can match.
	// seplen > 32 has no specialized scalar loop and must
	// use lsx_tail; jumping to scalar_dispatch would
	// re-enter the SIMD path via len_gt_or_eq_33 and loop
	// forever.
	ADDV	$-1, R19
	BNE	R19, lsx_cand
	ADDV	$1, R13, R4
	MOVV	$33, R15
	BGE	R7, R15, lsx_tail
	JMP	scalar_dispatch
lsx_advance:
	ADDV	$16, R4
	JMP	lsx_scan

	// Fewer than 16 start positions remain: scalar scan of the tail.
	// R10 is reloaded with sep[0]; verify uses R16/R17 so R10 survives.
lsx_tail:
	MOVBU	(R6), R10
lsx_tail_loop:
	BLT	R8, R4, not_found
	MOVBU	(R4), R11
	ADDV	$1, R4
	BNE	R10, R11, lsx_tail_loop
	MOVV	$1, R15
lsx_tail_verify:
	BEQ	R15, R7, found
	MOVBU	(R6)(R15), R16
	ADDV	R15, R4, R17
	MOVBU	-1(R17), R17
	BNE	R16, R17, lsx_tail_loop
	ADDV	$1, R15
	JMP	lsx_tail_verify

lasx_index:
	MOVBU	(R6), R10
	XVMOVQ	R10, X0.B32		// broadcast sep[0]
	ADDV	$-1, R7, R10
	MOVBU	(R6)(R10), R11
	XVMOVQ	R11, X1.B32		// broadcast sep[seplen-1]
	ADDV	$-31, R8, R12

lasx_scan:
	BLT	R12, R4, lsx_tail	// tail shared with LSX path
	XVMOVQ	(R4), X2
	XVSEQB	X2, X0, X2
	XVMOVQ	(R4)(R10), X3
	XVSEQB	X3, X1, X3
	XVANDV	X3, X2, X2
	XVMSKLTZB	X2, X4
	// xvmskltz.b: bits[15:0] = low lane, bits[143:128] = high lane
	XVMOVQ	X4.V[0], R14
	XVMOVQ	X4.V[2], R15
	SLLV	$16, R15
	OR	R15, R14		// 32-bit candidate mask
	BEQ	R14, lasx_advance
	MOVV	$8, R19			// candidate budget per block

lasx_cand:
	CTZV	R14, R15
	ADDV	R4, R15, R13
	BLT	R8, R13, not_found
	MOVV	$8, R15
	BLT	R7, R15, lasx_verify_bytes
	MOVV	(R6), R16
	MOVV	(R13), R17
	BNE	R16, R17, lasx_cand_next
	ADDV	$-8, R7, R18
	MOVV	$8, R15
lasx_verify_loop:
	BGE	R15, R18, lasx_verify_last
	MOVV	(R6)(R15), R16
	MOVV	(R13)(R15), R17
	BNE	R16, R17, lasx_cand_next
	ADDV	$8, R15
	JMP	lasx_verify_loop
lasx_verify_last:
	MOVV	(R6)(R18), R16
	MOVV	(R13)(R18), R17
	BNE	R16, R17, lasx_cand_next
	JMP	lasx_hit
lasx_verify_bytes:
	MOVV	R0, R15
lasx_verify_bloop:
	MOVBU	(R6)(R15), R16
	MOVBU	(R13)(R15), R17
	BNE	R16, R17, lasx_cand_next
	ADDV	$1, R15
	BNE	R15, R7, lasx_verify_bloop
lasx_hit:
	MOVV	R13, R4
	ADDV	$1, R4
	JMP	found
lasx_cand_next:
	ADDV	$-1, R14, R15
	AND	R15, R14
	BEQ	R14, lasx_advance
	ADDV	$-1, R19
	BNE	R19, lasx_cand
	ADDV	$1, R13, R4
	MOVV	$33, R15
	BGE	R7, R15, lsx_tail
	JMP	scalar_dispatch
lasx_advance:
	ADDV	$32, R4
	JMP	lasx_scan

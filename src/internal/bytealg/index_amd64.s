// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·Index(SB),NOSPLIT,$0-56
	MOVQ a_base+0(FP), DI
	MOVQ a_len+8(FP), DX
	MOVQ b_base+24(FP), R8
	MOVQ b_len+32(FP), AX
	MOVQ DI, R10
	LEAQ ret+48(FP), R11
	JMP  indexbody<>(SB)

TEXT ·IndexString(SB),NOSPLIT,$0-40
	MOVQ a_base+0(FP), DI
	MOVQ a_len+8(FP), DX
	MOVQ b_base+16(FP), R8
	MOVQ b_len+24(FP), AX
	MOVQ DI, R10
	LEAQ ret+32(FP), R11
	JMP  indexbody<>(SB)

// AX: length of string, that we are searching for
// DX: length of string, in which we are searching
// DI: pointer to string, in which we are searching
// R8: pointer to string, that we are searching for
// R11: address, where to put return value
// Note: We want len in DX and AX, because PCMPESTRI implicitly consumes them
TEXT indexbody<>(SB),NOSPLIT,$0
	CMPQ AX, DX
	JA fail
	CMPQ DX, $16
	JAE sse42
no_sse42:
	CMPQ AX, $2
	JA   _3_or_more
	MOVW (R8), R8
	LEAQ -1(DI)(DX*1), DX
loop2:
	MOVW (DI), SI
	CMPW SI,R8
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop2
	JMP fail
_3_or_more:
	CMPQ AX, $3
	JA   _4_or_more
	MOVW 1(R8), BX
	MOVW (R8), R8
	LEAQ -2(DI)(DX*1), DX
loop3:
	MOVW (DI), SI
	CMPW SI,R8
	JZ   partial_success3
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop3
	JMP fail
partial_success3:
	MOVW 1(DI), SI
	CMPW SI,BX
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop3
	JMP fail
_4_or_more:
	CMPQ AX, $4
	JA   _5_or_more
	MOVL (R8), R8
	LEAQ -3(DI)(DX*1), DX
loop4:
	MOVL (DI), SI
	CMPL SI,R8
	JZ   success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop4
	JMP fail
_5_or_more:
	CMPQ AX, $7
	JA   _8_or_more
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	MOVL -4(R8)(AX*1), BX
	MOVL (R8), R8
loop5to7:
	MOVL (DI), SI
	CMPL SI,R8
	JZ   partial_success5to7
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop5to7
	JMP fail
partial_success5to7:
	MOVL -4(AX)(DI*1), SI
	CMPL SI,BX
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop5to7
	JMP fail
_8_or_more:
	CMPQ AX, $8
	JA   _9_or_more
	MOVQ (R8), R8
	LEAQ -7(DI)(DX*1), DX
loop8:
	MOVQ (DI), SI
	CMPQ SI,R8
	JZ   success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop8
	JMP fail
_9_or_more:
	CMPQ AX, $15
	JA   _16_or_more
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	MOVQ -8(R8)(AX*1), BX
	MOVQ (R8), R8
loop9to15:
	MOVQ (DI), SI
	CMPQ SI,R8
	JZ   partial_success9to15
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop9to15
	JMP fail
partial_success9to15:
	MOVQ -8(AX)(DI*1), SI
	CMPQ SI,BX
	JZ success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop9to15
	JMP fail
_16_or_more:
	CMPQ AX, $16
	JA   _17_or_more
	MOVOU (R8), X1
	LEAQ -15(DI)(DX*1), DX
loop16:
	MOVOU (DI), X2
	PCMPEQB X1, X2
	PMOVMSKB X2, SI
	CMPQ  SI, $0xffff
	JE   success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop16
	JMP fail
_17_or_more:
	CMPQ AX, $31
	JA   _32_or_more
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	MOVOU -16(R8)(AX*1), X0
	MOVOU (R8), X1
loop17to31:
	MOVOU (DI), X2
	PCMPEQB X1,X2
	PMOVMSKB X2, SI
	CMPQ  SI, $0xffff
	JE   partial_success17to31
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop17to31
	JMP fail
partial_success17to31:
	MOVOU -16(AX)(DI*1), X3
	PCMPEQB X0, X3
	PMOVMSKB X3, SI
	CMPQ  SI, $0xffff
	JE success
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop17to31
	JMP fail
// We can get here only when AVX2 is enabled and cutoff for indexShortStr is set to 63
// So no need to check cpuid
_32_or_more:
	CMPQ AX, $32
	JA   _33_to_63
	VMOVDQU (R8), Y1
	LEAQ -31(DI)(DX*1), DX
loop32:
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPMOVMSKB Y3, SI
	CMPL  SI, $0xffffffff
	JE   success_avx2
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop32
	JMP fail_avx2
_33_to_63:
	LEAQ 1(DI)(DX*1), DX
	SUBQ AX, DX
	VMOVDQU -32(R8)(AX*1), Y0
	VMOVDQU (R8), Y1
loop33to63:
	VMOVDQU (DI), Y2
	VPCMPEQB Y1, Y2, Y3
	VPMOVMSKB Y3, SI
	CMPL  SI, $0xffffffff
	JE   partial_success33to63
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop33to63
	JMP fail_avx2
partial_success33to63:
	VMOVDQU -32(AX)(DI*1), Y3
	VPCMPEQB Y0, Y3, Y4
	VPMOVMSKB Y4, SI
	CMPL  SI, $0xffffffff
	JE success_avx2
	ADDQ $1,DI
	CMPQ DI,DX
	JB loop33to63
fail_avx2:
	VZEROUPPER
fail:
	MOVQ $-1, (R11)
	RET
success_avx2:
	VZEROUPPER
	JMP success
evex_prefix_match:
	// SIMD prefix matching
	// The algorithm use first 2 byte of needle as prefix, then
	// 1. bulk compare with haystack to see any prefix matching
	// 2. if found, compare with last byte to filter out unmatch tail quickly
	// 3. if still match candicates remains, do full match
	// (evex mask loading for data len < 32)

	// R9 is the safe pointer will not cross haystack boundary.
	// R12 is haystack start pointer for vec checking.
	LEAQ -32(DI)(DX*1), R9
	SUBQ AX, R9
	MOVQ DI, R12
	// Register allocation for needle:
	// Y1: 1st byte broadcast
	// Y2: 2nd byte broadcast
	// Y3: last byte broadcast
	// Y4: first 32 bytes of needle
	// Y5: needle[-32:] if have
	// K1: load mask for needle (ne_len <= 32)
	MOVBLZX (R8), BX
	MOVBLZX 1(R8), CX
	MOVBLZX -1(R8)(AX*1), SI
	VPBROADCASTB BX, Y1
	VPBROADCASTB CX, Y2
	VPBROADCASTB SI, Y3
	// Depends on needle length, we have 2 variants of full match:
	// * len <= 32: mask loading, cmp to single YMM register
	// * 64 > len > 32: overlap loading, cmp to 2 YMM register
	CMPQ AX, $32
	JA load_needle_overlap
	// load_mask = (-1L) >> (32 - ne_len)
	MOVL $0xffffffff, BX
	MOVL $32, CX
	SUBL AX, CX
	SHRL CX, BX
	KMOVD BX, K1
	// K1 is the load mask for needle.
	VMOVDQU8.Z (R8), K1, Y4
vec_loop_entry:
	// R12 -= 32 is for init the vec_loop.
	SUBQ $32, R12
vec_loop:
	ADDQ $32, R12
	CMPQ R12, R9
	JA tail_handler
	VPCMPEQB (R12), Y1, Y6
	VPCMPEQB 1(R12), Y2, Y7
	VPAND Y6, Y7, Y6
	VPMOVMSKB Y6, SI
	// Check if prefix match, otherwise, next loop.
	TESTL SI, SI
	JE    vec_loop
	// Filter out unmatch tail.
	VPCMPEQB -1(R12)(AX*1), Y3, Y7
	VPAND Y6, Y7, Y6
	VPMOVMSKB Y6, SI
	TESTL SI, SI
	JE    vec_loop
next_pair_index:
	// Match found, extract first match index.
	BSFL SI, BX
	LEAQ (R12)(BX*1), DI
	CMPQ AX, $32
	JA  full_match_overlap
	// Test if full match.
	VMOVDQU8.Z (DI), K1, Y6
	VPCMPEQB Y6, Y4, Y7
	VPMOVMSKB Y7, CX
	// Use full match since unused Y6, Y4 bytes are all zeros.
	CMPL  CX, $0xffffffff
	JE    success_avx2
prepare_next_pair:
	// False matching, iter to next pair index.
	LEAL -1(SI), CX
	ANDL CX, SI
	JNE  next_pair_index
	// No more matched bits, back to loop.
	JMP  vec_loop
full_match_overlap:
	VPCMPEQB (DI), Y4, Y6
	VPCMPEQB -32(DI)(AX*1), Y5, Y7
	VPAND Y6, Y7, Y7
	VPMOVMSKB Y7, CX
	CMPL  CX, $0xffffffff
	JE    success_avx2
	JMP   prepare_next_pair
tail_handler:
	// Save possible matching mask to K2
	// last possible matching end_pos = hs_end - ne_len + 1
	// while R9 = (hs_end - ne_len) - 32
	// remain potential match_count = end_pos - R12
	//                              = hs_end - ne_len + 1- R12 = R9 - R12 + 33
	// mask = (-1L) >> (32 - match_count) = (-1L) >> (R12 - R9 - 1)
	MOVL $0xffffffff, BX
	LEAQ -1(R12), CX
	SUBQ R9, CX
	SHRL CX, BX
	KMOVD BX, K2
	// Prefix mathcing.
	VMOVDQU8.Z (R12), K2, Y6
	VMOVDQU8.Z 1(R12), K2, Y7
	VPCMPEQB Y6, Y1, Y6
	VPCMPEQB Y7, Y2, Y7
	VPAND Y6, Y7, Y6
	VPMOVMSKB Y6, SI
	TESTL SI, SI
	JE    fail_avx2
	// Filter out unmatch tail.
	VMOVDQU8.Z -1(R12)(AX*1), K2, Y7
	VPCMPEQB Y7, Y3, Y7
	VPAND Y6, Y7, Y6
	VPMOVMSKB Y6, SI
	// Discard false match that out of range.
	ANDL BX, SI
	TESTL SI, SI
	JE    fail_avx2
next_pair_index1:
	// Match found, extract first index.
	BSFL SI, BX
	LEAQ (R12)(BX*1), DI
	CMPQ AX, $32
	JA  full_match_overlap1
	// Test if full match.
	VMOVDQU8.Z (DI), K1, Y6
	VPCMPEQB Y6, Y4, Y7
	VPMOVMSKB Y7, CX
	CMPL  CX, $0xffffffff
	JE    success_avx2
prepare_next_pair1:
	// False matching, iter to next pair index.
	LEAL -1(SI), CX
	ANDL CX, SI
	JNE  next_pair_index1
	// No match any more, fail.
	JMP  fail_avx2
full_match_overlap1:
	VPCMPEQB (DI), Y4, Y6
	VPCMPEQB -32(DI)(AX*1), Y5, Y7
	VPAND Y6, Y7, Y7
	VPMOVMSKB Y7, CX
	CMPL  CX, $0xffffffff
	JE    success_avx2
	JMP   prepare_next_pair1
load_needle_overlap:
	// Needle length > 32, use overlap loading.
	VMOVDQU (R8), Y4
	VMOVDQU -32(R8)(AX*1), Y5
	JMP  vec_loop_entry
sse42:
	CMPB internal∕cpu·X86+const_offsetX86HasSSE42(SB), $1
	JNE no_sse42
	MOVBLZX internal∕cpu·X86+const_offsetX86HasAVX512BW(SB), BX
	TESTB internal∕cpu·X86+const_offsetX86HasAVX512VL(SB), BX
	JNZ evex_prefix_match
	CMPQ AX, $12
	// PCMPESTRI is slower than normal compare,
	// so using it makes sense only if we advance 4+ bytes per compare
	// This value was determined experimentally and is the ~same
	// on Nehalem (first with SSE42) and Haswell.
	JAE _9_or_more
	LEAQ 16(R8), SI
	TESTW $0xff0, SI
	JEQ no_sse42
	MOVOU (R8), X1
	LEAQ -15(DI)(DX*1), SI
	MOVQ $16, R9
	SUBQ AX, R9 // We advance by 16-len(sep) each iteration, so precalculate it into R9
loop_sse42:
	// 0x0c means: unsigned byte compare (bits 0,1 are 00)
	// for equality (bits 2,3 are 11)
	// result is not masked or inverted (bits 4,5 are 00)
	// and corresponds to first matching byte (bit 6 is 0)
	PCMPESTRI $0x0c, (DI), X1
	// CX == 16 means no match,
	// CX > R9 means partial match at the end of the string,
	// otherwise sep is at offset CX from X1 start
	CMPQ CX, R9
	JBE sse42_success
	ADDQ R9, DI
	CMPQ DI, SI
	JB loop_sse42
	PCMPESTRI $0x0c, -1(SI), X1
	CMPQ CX, R9
	JA fail
	LEAQ -1(SI), DI
sse42_success:
	ADDQ CX, DI
success:
	SUBQ R10, DI
	MOVQ DI, (R11)
	RET

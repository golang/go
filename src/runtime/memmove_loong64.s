// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// See memmove Go doc for important implementation constraints.

// Register map
//
// to		R4
// from		R5
// n(aka count)	R6
// to-end	R7
// from-end	R8
// data		R11-R18
// tmp		R9

// Algorithm:
//
// Memory alignment check is only performed for copy size greater
// than 64 bytes to minimize overhead.
//
// when copy size <= 64 bytes, jump to label tail, according to the
// copy size to select the appropriate case and copy directly.
// Based on the common memory access instructions of loong64, the
// currently implemented cases are:
// move_0, move_1, move_2, move_3, move_4, move_5through7, move_8,
// move_9through16, move_17through32, move_33through64
//
// when copy size > 64 bytes, use the destination-aligned copying,
// adopt the following strategy to copy in 3 parts:
// 1. Head: do the memory alignment
// 2. Body: a 64-byte loop structure
// 3. Tail: processing of the remaining part (<= 64 bytes)
//
// forward:
//
//    Dst           NewDst                           Dstend
//     |               |<----count after correction---->|
//     |<-------------count before correction---------->|
//     |<--8-(Dst&7)-->|               |<---64 bytes--->|
//     +------------------------------------------------+
//     |   Head        |      Body     |      Tail      |
//     +---------------+---------------+----------------+
//    NewDst = Dst - (Dst & 7) + 8
//    count = count - 8 + (Dst & 7)
//    Src = Src - (Dst & 7) + 8
//
// backward:
//
//    Dst                             NewDstend          Dstend
//     |<-----count after correction------>|                |
//     |<------------count before correction--------------->|
//     |<---64 bytes--->|                  |<---Dstend&7--->|
//     +----------------------------------------------------+
//     |   Tail         |      Body        |      Head      |
//     +----------------+------------------+----------------+
//    NewDstend = Dstend - (Dstend & 7)
//    count = count - (Dstend & 7)
//    Srcend = Srcend - (Dstend & 7)

// func memmove(to, from unsafe.Pointer, n uintptr)
TEXT runtime·memmove<ABIInternal>(SB), NOSPLIT|NOFRAME, $0-24
	BEQ	R4, R5, move_0
	BEQ	R6, move_0

	ADDV	R4, R6, R7	// to-end pointer
	ADDV	R5, R6, R8	// from-end pointer

// copy size <= 64 bytes, copy directly, not check aligned
tail:
	// < 2 bytes
	SGTU	$2, R6, R9
	BNE	R9, move_1

	// < 3 bytes
	SGTU	$3, R6, R9
	BNE	R9, move_2

	// < 4 bytes
	SGTU	$4, R6, R9
	BNE	R9, move_3

	// < 5 bytes
	SGTU	$5, R6, R9
	BNE	R9, move_4

	// >= 5 bytes and < 8 bytes
	SGTU	$8, R6, R9
	BNE	R9, move_5through7

	// < 9 bytes
	SGTU	$9, R6, R9
	BNE	R9, move_8

	// >= 9 bytes and < 17 bytes
	SGTU	$17, R6, R9
	BNE	R9, move_9through16

	// >= 17 bytes and < 33 bytes
	SGTU	$33, R6, R9
	BNE	R9, move_17through32

	// >= 33 bytes and < 65 bytes
	SGTU	$65, R6, R9
	BNE	R9, move_33through64

	// >= 65 bytes and < 256 bytes
	SGTU	$256, R6, R9
	BNE	R9, move_large

	// >= 256
	JMP	lasx_move_large

move_0:
	RET

move_1:
	MOVB	(R5), R11
	MOVB	R11, (R4)
	RET
move_2:
	MOVH	(R5), R11
	MOVH	R11, (R4)
	RET
move_3:
	MOVH	(R5), R11
	MOVB	-1(R8), R12
	MOVH	R11, (R4)
	MOVB	R12, -1(R7)
	RET
move_4:
	MOVW	(R5), R11
	MOVW	R11, (R4)
	RET
move_5through7:
	MOVW	(R5), R11
	MOVW	-4(R8), R12
	MOVW	R11, (R4)
	MOVW	R12, -4(R7)
	RET
move_8:
	MOVV	(R5), R11
	MOVV	R11, (R4)
	RET
move_9through16:
	MOVV	(R5), R11
	MOVV	-8(R8), R12
	MOVV	R11, (R4)
	MOVV	R12, -8(R7)
	RET
move_17through32:
	MOVV	(R5), R11
	MOVV	8(R5), R12
	MOVV	-16(R8), R13
	MOVV	-8(R8), R14
	MOVV	R11, (R4)
	MOVV	R12, 8(R4)
	MOVV	R13, -16(R7)
	MOVV	R14, -8(R7)
	RET
move_33through64:
	MOVV	(R5), R11
	MOVV	8(R5), R12
	MOVV	16(R5), R13
	MOVV	24(R5), R14
	MOVV	-32(R8), R15
	MOVV	-24(R8), R16
	MOVV	-16(R8), R17
	MOVV	-8(R8), R18
	MOVV	R11, (R4)
	MOVV	R12, 8(R4)
	MOVV	R13, 16(R4)
	MOVV	R14, 24(R4)
	MOVV	R15, -32(R7)
	MOVV	R16, -24(R7)
	MOVV	R17, -16(R7)
	MOVV	R18, -8(R7)
	RET

move_large:
	// if (dst > src) && (dst < (src + count))
	//    regarded as memory overlap
	//    jump to backward
	// else
	//    jump to forward
	BGEU	R5, R4, forward
	ADDV	R5, R6, R10
	BLTU	R4, R10, backward
forward:
	AND	$7, R4, R9	// dst & 7
	BEQ	R9, forward_move_64loop
forward_unaligned:
	MOVV	$8, R10
	SUBV	R9, R10	// head = 8 - (dst & 7)
	MOVV	(R5), R11
	SUBV	R10, R6	// newcount = count - (8 - (dst & 7))
	ADDV	R10, R5	// newsrc = src + (8 - (dst & 7))
	MOVV	(R5), R12
	MOVV	R11, (R4)
	ADDV	R10, R4	// newdst = dst + (8 - (dst & 7))
	MOVV	R12, (R4)
	SUBV	$8, R6
	ADDV	$8, R4
	ADDV	$8, R5
	SGTU	$65, R6, R9
	BNE	R9, move_33through64
forward_move_64loop:
	SUBV	$64, R6
	SGTU	$64, R6, R9
	MOVV	(R5), R11
	MOVV	8(R5), R12
	MOVV	16(R5), R13
	MOVV	24(R5), R14
	MOVV	32(R5), R15
	MOVV	40(R5), R16
	MOVV	48(R5), R17
	MOVV	56(R5), R18
	MOVV	R11, (R4)
	MOVV	R12, 8(R4)
	MOVV	R13, 16(R4)
	MOVV	R14, 24(R4)
	MOVV	R15, 32(R4)
	MOVV	R16, 40(R4)
	MOVV	R17, 48(R4)
	MOVV	R18, 56(R4)
	ADDV	$64, R5
	ADDV	$64, R4
	BEQ	R9, forward_move_64loop
	// 0 < remaining_length < 64
	BNE	R6, tail
	RET

// The backward copy algorithm is the same as the forward
// copy, except for the direction.
backward:
	AND	$7, R7, R9	// dstend & 7
	BEQ	R9, backward_move_64loop
backward_unaligned:
	MOVV	-8(R8), R11
	SUBV	R9, R6	// newcount = count - (dstend & 7)
	SUBV	R9, R8	// newsrcend = srcend - (dstend & 7)
	MOVV	-8(R8), R12
	MOVV	R11, -8(R7)
	SUBV	R9, R7	// newdstend = dstend - (dstend & 7)
	MOVV	R12, -8(R7)
	SUBV	$8, R6
	SUBV	$8, R7
	SUBV	$8, R8
	SGTU    $65, R6, R9
	BNE     R9, move_33through64
backward_move_64loop:
	SUBV	$64, R6
	SGTU	$64, R6, R9
	MOVV	-8(R8), R11
	MOVV	-16(R8), R12
	MOVV	-24(R8), R13
	MOVV	-32(R8), R14
	MOVV	-40(R8), R15
	MOVV	-48(R8), R16
	MOVV	-56(R8), R17
	MOVV	-64(R8), R18
	MOVV	R11, -8(R7)
	MOVV	R12, -16(R7)
	MOVV	R13, -24(R7)
	MOVV	R14, -32(R7)
	MOVV	R15, -40(R7)
	MOVV	R16, -48(R7)
	MOVV	R17, -56(R7)
	MOVV	R18, -64(R7)
	SUBV	$64, R7
	SUBV	$64, R8
	BEQ	R9, backward_move_64loop
	// 0 < remaining_length < 64
	BNE	R6, tail
	RET

// use simd 128 instructions to implement memmove
// n >= 256 bytes, check 16-byte alignment
lsx_move_large:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R9
	BEQ	R9, move_large

	// if (dst > src) && (dst < (src + count))
	//    regarded as memory overlap
	//    jump to lsx_backward
	// else
	//    jump to lsx_forward
	BGEU	R5, R4, lsx_forward
	ADDV	R5, R6, R10
	BLTU	R4, R10, lsx_backward
lsx_forward:
	AND	$15, R4, R9	// dst & 15
	BEQ	R9, lsx_forward_move_128
lsx_forward_unaligned:
	MOVV	$16, R10
	SUBV	R9, R10	// head = 16 - (dst & 15)
	VMOVQ	(R5), V0
	SUBV	R10, R6	// newcount = count - (16 - (dst & 15))
	ADDV	R10, R5	// newsrc = src + (16 - (dst & 15))
	VMOVQ	(R5), V1
	VMOVQ	V0, (R4)
	ADDV	R10, R4	// newdst = dst + (16 - (dst & 15))
	VMOVQ	V1, (R4)
	SUBV	$16, R6
	ADDV	$16, R4
	ADDV	$16, R5
lsx_forward_move_128:
	SGTU	$128, R6, R9
	BNE	R9, lsx_forward_move_32
lsx_forward_move_128loop:
	SUBV	$128, R6
	SGTU	$128, R6, R9
	VMOVQ	0(R5), V0
	VMOVQ	16(R5), V1
	VMOVQ	32(R5), V2
	VMOVQ	48(R5), V3
	VMOVQ	64(R5), V4
	VMOVQ	80(R5), V5
	VMOVQ	96(R5), V6
	VMOVQ	112(R5), V7
	VMOVQ	V0, 0(R4)
	VMOVQ	V1, 16(R4)
	VMOVQ	V2, 32(R4)
	VMOVQ	V3, 48(R4)
	VMOVQ	V4, 64(R4)
	VMOVQ	V5, 80(R4)
	VMOVQ	V6, 96(R4)
	VMOVQ	V7, 112(R4)
	ADDV	$128, R5
	ADDV	$128, R4
	BEQ	R9, lsx_forward_move_128loop
lsx_forward_move_32:
	SGTU	$32, R6, R9
	BNE	R9, lsx_forward_move_tail
lsx_forward_move_32loop:
	SUBV	$32, R6
	SGTU	$32, R6, R9
	VMOVQ	0(R5), V0
	VMOVQ	16(R5), V1
	VMOVQ	V0, 0(R4)
	VMOVQ	V1, 16(R4)
	ADDV	$32, R5
	ADDV	$32, R4
	BEQ	R9, lsx_forward_move_32loop
lsx_forward_move_tail:
	// 0 < remaining_length < 64
	BNE	R6, tail
	RET

lsx_backward:
	AND	$15, R7, R9	// dstend & 15
	BEQ	R9, lsx_backward_move_128
lsx_backward_unaligned:
	VMOVQ	-16(R8), V0
	SUBV	R9, R6	// newcount = count - (dstend & 15)
	SUBV	R9, R8	// newsrcend = srcend - (dstend & 15)
	VMOVQ	-16(R8), V1
	VMOVQ	V0, -16(R7)
	SUBV	R9, R7	// newdstend = dstend - (dstend & 15)
	VMOVQ	V1, -16(R7)
	SUBV	$16, R6
	SUBV	$16, R7
	SUBV	$16, R8
lsx_backward_move_128:
	SGTU    $128, R6, R9
	BNE     R9, lsx_backward_move_32
lsx_backward_move_128loop:
	SUBV	$128, R6
	SGTU	$128, R6, R9
	VMOVQ	-16(R8), V0
	VMOVQ	-32(R8), V1
	VMOVQ	-48(R8), V2
	VMOVQ	-64(R8), V3
	VMOVQ	-80(R8), V4
	VMOVQ	-96(R8), V5
	VMOVQ	-112(R8), V6
	VMOVQ	-128(R8), V7
	VMOVQ	V0, -16(R7)
	VMOVQ	V1, -32(R7)
	VMOVQ	V2, -48(R7)
	VMOVQ	V3, -64(R7)
	VMOVQ	V4, -80(R7)
	VMOVQ	V5, -96(R7)
	VMOVQ	V6, -112(R7)
	VMOVQ	V7, -128(R7)
	SUBV	$128, R8
	SUBV	$128, R7
	BEQ	R9, lsx_backward_move_128loop
lsx_backward_move_32:
	SGTU    $32, R6, R9
	BNE     R9, lsx_backward_move_tail
lsx_backward_move_32loop:
	SUBV	$32, R6
	SGTU	$32, R6, R9
	VMOVQ	-16(R8), V0
	VMOVQ	-32(R8), V1
	VMOVQ	V0, -16(R7)
	VMOVQ	V1, -32(R7)
	SUBV	$32, R8
	SUBV	$32, R7
	BEQ	R9, lsx_backward_move_32loop
lsx_backward_move_tail:
	// 0 < remaining_length < 64
	BNE	R6, tail
	RET

// use simd 256 instructions to implement memmove
// n >= 256 bytes, check 32-byte alignment
lasx_move_large:
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R9
	BEQ	R9, lsx_move_large

	// if (dst > src) && (dst < (src + count))
	//    regarded as memory overlap
	//    jump to lasx_backward
	// else
	//    jump to lasx_forward
	BGEU	R5, R4, lasx_forward
	ADDV	R5, R6, R10
	BLTU	R4, R10, lasx_backward
lasx_forward:
	AND	$31, R4, R9	// dst & 31
	BEQ	R9, lasx_forward_move_256
lasx_forward_unaligned:
	MOVV	$32, R10
	SUBV	R9, R10	// head = 32 - (dst & 31)
	XVMOVQ	(R5), X0
	SUBV	R10, R6	// newcount = count - (32 - (dst & 31))
	ADDV	R10, R5	// newsrc = src + (32 - (dst & 31))
	XVMOVQ	(R5), X1
	XVMOVQ	X0, (R4)
	ADDV	R10, R4	// newdst = dst + (32 - (dst & 31))
	XVMOVQ	X1, (R4)
	SUBV	$32, R6
	ADDV	$32, R4
	ADDV	$32, R5
lasx_forward_move_256:
	SGTU	$256, R6, R9
	BNE	R9, lasx_forward_move_64
lasx_forward_move_256loop:
	SUBV	$256, R6
	SGTU	$256, R6, R9
	XVMOVQ	0(R5), X0
	XVMOVQ	32(R5), X1
	XVMOVQ	64(R5), X2
	XVMOVQ	96(R5), X3
	XVMOVQ	128(R5), X4
	XVMOVQ	160(R5), X5
	XVMOVQ	192(R5), X6
	XVMOVQ	224(R5), X7
	XVMOVQ	X0, 0(R4)
	XVMOVQ	X1, 32(R4)
	XVMOVQ	X2, 64(R4)
	XVMOVQ	X3, 96(R4)
	XVMOVQ	X4, 128(R4)
	XVMOVQ	X5, 160(R4)
	XVMOVQ	X6, 192(R4)
	XVMOVQ	X7, 224(R4)
	ADDV	$256, R5
	ADDV	$256, R4
	BEQ	R9, lasx_forward_move_256loop
lasx_forward_move_64:
	SGTU	$64, R6, R9
	BNE	R9, lasx_forward_move_tail
lasx_forward_move_64loop:
	SUBV	$64, R6
	SGTU	$64, R6, R9
	XVMOVQ	(R5), X0
	XVMOVQ	32(R5), X1
	XVMOVQ	X0, (R4)
	XVMOVQ	X1, 32(R4)
	ADDV	$64, R5
	ADDV	$64, R4
	BEQ	R9, lasx_forward_move_64loop
lasx_forward_move_tail:
	// 0 < remaining_length < 64
	BNE	R6, tail
	RET

lasx_backward:
	AND	$31, R7, R9	// dstend & 31
	BEQ	R9, lasx_backward_move_256
lasx_backward_unaligned:
	XVMOVQ	-32(R8), X0
	SUBV	R9, R6	// newcount = count - (dstend & 31)
	SUBV	R9, R8	// newsrcend = srcend - (dstend & 31)
	XVMOVQ	-32(R8), X1
	XVMOVQ	X0, -32(R7)
	SUBV	R9, R7	// newdstend = dstend - (dstend & 31)
	XVMOVQ	X1, -32(R7)
	SUBV	$32, R6
	SUBV	$32, R7
	SUBV	$32, R8
lasx_backward_move_256:
	SGTU    $256, R6, R9
	BNE     R9, lasx_backward_move_64
lasx_backward_move_256loop:
	SUBV	$256, R6
	SGTU	$256, R6, R9
	XVMOVQ	-32(R8), X0
	XVMOVQ	-64(R8), X1
	XVMOVQ	-96(R8), X2
	XVMOVQ	-128(R8), X3
	XVMOVQ	-160(R8), X4
	XVMOVQ	-192(R8), X5
	XVMOVQ	-224(R8), X6
	XVMOVQ	-256(R8), X7
	XVMOVQ	X0, -32(R7)
	XVMOVQ	X1, -64(R7)
	XVMOVQ	X2, -96(R7)
	XVMOVQ	X3, -128(R7)
	XVMOVQ	X4, -160(R7)
	XVMOVQ	X5, -192(R7)
	XVMOVQ	X6, -224(R7)
	XVMOVQ	X7, -256(R7)
	SUBV	$256, R8
	SUBV	$256, R7
	BEQ	R9, lasx_backward_move_256loop
lasx_backward_move_64:
	SGTU	$64, R6, R9
	BNE     R9, lasx_backward_move_tail
lasx_backward_move_64loop:
	SUBV	$64, R6
	SGTU	$64, R6, R9
	XVMOVQ	-32(R8), X0
	XVMOVQ	-64(R8), X1
	XVMOVQ	X0, -32(R7)
	XVMOVQ	X1, -64(R7)
	SUBV	$64, R8
	SUBV	$64, R7
	BEQ	R9, lasx_backward_move_64loop
lasx_backward_move_tail:
	// 0 < remaining_length < 64
	BNE	R6, tail
	RET

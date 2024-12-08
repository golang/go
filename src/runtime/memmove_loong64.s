// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

	ADDV	R4, R6, R7 // to-end pointer
	ADDV	R5, R6, R8 // from-end pointer

tail:
	//copy size <= 64 bytes, copy directly, not check aligned

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

	// if (dst > src) && (dst < src + count), regarded as memory
	// overlap, jump to backward
	// else, jump to forward
	BGEU	R5, R4, forward
	ADDV	R5, R6, R10
	BLTU	R4, R10, backward

forward:
	AND	$7, R4, R9	// dst & 7
	BEQ	R9, body
head:
	MOVV	$8, R10
	SUBV	R9, R10		// head = 8 - (dst & 7)
	MOVB	(R5), R11
	SUBV	$1, R10
	ADDV	$1, R5
	MOVB	R11, (R4)
	ADDV	$1, R4
	BNE	R10, -5(PC)
	ADDV	R9, R6
	ADDV	$-8, R6		// newcount = count + (dst & 7) - 8
	// if newcount < 65 bytes, use move_33through64 to copy is enough
	SGTU	$65, R6, R9
	BNE	R9, move_33through64

body:
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
	ADDV	$-64, R6
	ADDV	$64, R4
	ADDV	$64, R5
	SGTU	$64, R6, R9
	// if the remaining part >= 64 bytes, jmp to body
	BEQ	R9, body
	// if the remaining part == 0 bytes, use move_0 to return
	BEQ	R6, move_0
	// if the remaining part in (0, 63] bytes, jmp to tail
	JMP	tail

// The backward copy algorithm is the same as the forward copy,
// except for the direction.
backward:
	AND	$7, R7, R9	 // dstend & 7
	BEQ	R9, b_body
b_head:
	MOVV	-8(R8), R11
	SUBV	R9, R6		// newcount = count - (dstend & 7)
	SUBV	R9, R8		// newsrcend = srcend - (dstend & 7)
	MOVV	-8(R8), R12
 	MOVV	R11, -8(R7)
	SUBV	R9, R7		// newdstend = dstend - (dstend & 7)
 	MOVV	R12, -8(R7)
	SUBV	$8, R6
	SUBV	$8, R7
	SUBV	$8, R8
	SGTU    $65, R6, R9
	BNE     R9, move_33through64

b_body:
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
	ADDV	$-64, R6
	ADDV	$-64, R7
	ADDV	$-64, R8
	SGTU	$64, R6, R9
	BEQ	R9, b_body
	BEQ	R6, move_0
	JMP	tail

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

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// Register map
//
// R4: ptr
// R5: n
// R6: ptrend
// R7: tmp

// Algorithm:
//
// 1. when count <= 64 bytes, memory alignment check is omitted.
// The handling is divided into distinct cases based on the size
// of count: clr_0, clr_1, clr_2, clr_3, clr_4, clr_5through7,
// clr_8, clr_9through16, clr_17through32, and clr_33through64.
//
// 2. when count > 64 bytes, memory alignment check is performed.
// Unaligned bytes are processed first (that is, 8-(ptr&7)), and
// then a 64-byte loop is executed to zero out memory.
// When the number of remaining bytes not cleared is n < 64 bytes,
// a tail processing is performed, invoking the corresponding case
// based on the size of n.
//
//    ptr           newptr                           ptrend
//     |               |<----count after correction---->|
//     |<-------------count before correction---------->|
//     |<--8-(ptr&7)-->|               |<---64 bytes--->|
//     +------------------------------------------------+
//     |   Head        |      Body     |      Tail      |
//     +---------------+---------------+----------------+
//    newptr = ptr - (ptr & 7) + 8
//    count = count - 8 + (ptr & 7)

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers<ABIInternal>(SB),NOSPLIT,$0-16
	BEQ	R5, clr_0
	ADDV	R4, R5, R6

tail:
	// <=64 bytes, clear directly, not check aligned
	SGTU	$2, R5, R7
	BNE	R7, clr_1
	SGTU	$3, R5, R7
	BNE	R7, clr_2
	SGTU	$4, R5, R7
	BNE	R7, clr_3
	SGTU	$5, R5, R7
	BNE	R7, clr_4
	SGTU	$8, R5, R7
	BNE	R7, clr_5through7
	SGTU	$9, R5, R7
	BNE	R7, clr_8
	SGTU	$17, R5, R7
	BNE	R7, clr_9through16
	SGTU	$33, R5, R7
	BNE	R7, clr_17through32
	SGTU	$65, R5, R7
	BNE	R7, clr_33through64

	// n > 64 bytes, check aligned
	AND	$7, R4, R7
	BEQ	R7, body

head:
	MOVV	R0, (R4)
	SUBV	R7, R4
	ADDV	R7, R5
	ADDV	$8, R4	// newptr = ptr + (8 - (ptr & 7))
	SUBV	$8, R5	// newn = n - (8 - (ptr & 7))
	SGTU	$65, R5, R7
	BNE	R7, clr_33through64

body:
	MOVV	R0, (R4)
	MOVV	R0, 8(R4)
	MOVV	R0, 16(R4)
	MOVV	R0, 24(R4)
	MOVV	R0, 32(R4)
	MOVV	R0, 40(R4)
	MOVV	R0, 48(R4)
	MOVV	R0, 56(R4)
	ADDV	$-64, R5
	ADDV	$64, R4
	SGTU	$65, R5, R7
	BEQ	R7, body
	BEQ	R5, clr_0
	JMP	tail

clr_0:
	RET
clr_1:
	MOVB	R0, (R4)
	RET
clr_2:
	MOVH	R0, (R4)
	RET
clr_3:
	MOVH	R0, (R4)
	MOVB	R0, 2(R4)
	RET
clr_4:
	MOVW	R0, (R4)
	RET
clr_5through7:
	MOVW	R0, (R4)
	MOVW	R0, -4(R6)
	RET
clr_8:
	MOVV	R0, (R4)
	RET
clr_9through16:
	MOVV	R0, (R4)
	MOVV	R0, -8(R6)
	RET
clr_17through32:
	MOVV	R0, (R4)
	MOVV	R0, 8(R4)
	MOVV	R0, -16(R6)
	MOVV	R0, -8(R6)
	RET
clr_33through64:
	MOVV	R0, (R4)
	MOVV	R0, 8(R4)
	MOVV	R0, 16(R4)
	MOVV	R0, 24(R4)
	MOVV	R0, -32(R6)
	MOVV	R0, -24(R6)
	MOVV	R0, -16(R6)
	MOVV	R0, -8(R6)
	RET

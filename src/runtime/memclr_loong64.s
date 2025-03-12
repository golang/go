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
// 1. if lasx is enabled:
//        THRESHOLD = 256, ALIGNMENTS = 32, LOOPBLOCKS = 256,
//    else if lsx is enabled:
//        THRESHOLD = 128, ALIGNMENTS = 16, LOOPBLOCKS = 128,
//    else
//        THRESHOLD = 64, ALIGNMENTS = 8, LOOPBLOCKS = 64,
//
// 2. when 'count <= THRESHOLD' bytes, memory alignment check is omitted.
// The handling is divided into distinct cases based on the size of count:
//   a. clr_0, clr_1, clr_2, clr_3, clr_4, clr_5through7, clr_8,
//      clr_9through16, clr_17through32, clr_33through64,
//   b. lsx_clr_17through32, lsx_clr_33through64, lsx_clr_65through128,
//   c. lasx_clr_17through32, lasx_clr_33through64, lsx_clr_65through128,
//      lasx_clr_65through128, lasx_clr_129through256
//
// 3. when 'count > THRESHOLD' bytes, memory alignment check is performed. Unaligned
// bytes are processed first (that is, ALIGNMENTS - (ptr & (ALIGNMENTS-1))), and then
// a LOOPBLOCKS-byte loop is executed to zero out memory.
// When the number of remaining bytes not cleared is n < LOOPBLOCKS bytes, a tail
// processing is performed, invoking the corresponding case based on the size of n.
//
// example:
//    THRESHOLD = 64, ALIGNMENTS = 8, LOOPBLOCKS = 64
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

	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLASX(SB), R7
	BNE	R7, lasx_tail
	MOVBU	internal∕cpu·Loong64+const_offsetLOONG64HasLSX(SB), R7
	BNE	R7, lsx_tail

	SGTU	$33, R5, R7
	BNE	R7, clr_17through32
	SGTU	$65, R5, R7
	BNE	R7, clr_33through64
	JMP	clr_large

lasx_tail:
	// X0 = 0
	XVXORV	X0, X0, X0

	SGTU	$33, R5, R7
	BNE	R7, lasx_clr_17through32
	SGTU	$65, R5, R7
	BNE	R7, lasx_clr_33through64
	SGTU	$129, R5, R7
	BNE	R7, lasx_clr_65through128
	SGTU	$257, R5, R7
	BNE	R7, lasx_clr_129through256
	JMP	lasx_clr_large

lsx_tail:
	// V0 = 0
	VXORV	V0, V0, V0

	SGTU	$33, R5, R7
	BNE	R7, lsx_clr_17through32
	SGTU	$65, R5, R7
	BNE	R7, lsx_clr_33through64
	SGTU	$129, R5, R7
	BNE	R7, lsx_clr_65through128
	JMP	lsx_clr_large

	// use simd 256 instructions to implement memclr
	// n > 256 bytes, check 32-byte alignment
lasx_clr_large:
	AND	$31, R4, R7
	BEQ	R7, lasx_clr_256loop
	XVMOVQ	X0, (R4)
	SUBV	R7, R4
	ADDV	R7, R5
	SUBV	$32, R5 // newn = n - (32 - (ptr & 31))
	ADDV	$32, R4 // newptr = ptr + (32 - (ptr & 31))
	SGTU	$257, R5, R7
	BNE	R7, lasx_clr_129through256
lasx_clr_256loop:
	SUBV	$256, R5
	SGTU	$256, R5, R7
	XVMOVQ	X0, 0(R4)
	XVMOVQ	X0, 32(R4)
	XVMOVQ	X0, 64(R4)
	XVMOVQ	X0, 96(R4)
	XVMOVQ	X0, 128(R4)
	XVMOVQ	X0, 160(R4)
	XVMOVQ	X0, 192(R4)
	XVMOVQ	X0, 224(R4)
	ADDV	$256, R4
	BEQ	R7, lasx_clr_256loop

	// remaining_length is 0
	BEQ	R5, clr_0

	// 128 < remaining_length < 256
	SGTU	$129, R5, R7
	BEQ	R7, lasx_clr_129through256

	// 64 < remaining_length <= 128
	SGTU	$65, R5, R7
	BEQ	R7, lasx_clr_65through128

	// 32 < remaining_length <= 64
	SGTU	$33, R5, R7
	BEQ	R7, lasx_clr_33through64

	// 16 < remaining_length <= 32
	SGTU	$17, R5, R7
	BEQ	R7, lasx_clr_17through32

	// 0 < remaining_length <= 16
	JMP	tail

	// use simd 128 instructions to implement memclr
	// n > 128 bytes, check 16-byte alignment
lsx_clr_large:
	// check 16-byte alignment
	AND	$15, R4, R7
	BEQ	R7, lsx_clr_128loop
	VMOVQ	V0, (R4)
	SUBV	R7, R4
	ADDV	R7, R5
	SUBV	$16, R5 // newn = n - (16 - (ptr & 15))
	ADDV	$16, R4 // newptr = ptr + (16 - (ptr & 15))
	SGTU	$129, R5, R7
	BNE	R7, lsx_clr_65through128
lsx_clr_128loop:
	SUBV	$128, R5
	SGTU	$128, R5, R7
	VMOVQ	V0, 0(R4)
	VMOVQ	V0, 16(R4)
	VMOVQ	V0, 32(R4)
	VMOVQ	V0, 48(R4)
	VMOVQ	V0, 64(R4)
	VMOVQ	V0, 80(R4)
	VMOVQ	V0, 96(R4)
	VMOVQ	V0, 112(R4)
	ADDV	$128, R4
	BEQ	R7, lsx_clr_128loop

	// remaining_length is 0
	BEQ	R5, clr_0

	// 64 < remaining_length <= 128
	SGTU	$65, R5, R7
	BEQ	R7, lsx_clr_65through128

	// 32 < remaining_length <= 64
	SGTU	$33, R5, R7
	BEQ	R7, lsx_clr_33through64

	// 16 < remaining_length <= 32
	SGTU	$17, R5, R7
	BEQ	R7, lsx_clr_17through32

	// 0 < remaining_length <= 16
	JMP	tail

	// use general instructions to implement memclr
	// n > 64 bytes, check 16-byte alignment
clr_large:
	AND	$7, R4, R7
	BEQ	R7, clr_64loop
	MOVV	R0, (R4)
	SUBV	R7, R4
	ADDV	R7, R5
	ADDV	$8, R4	// newptr = ptr + (8 - (ptr & 7))
	SUBV	$8, R5	// newn = n - (8 - (ptr & 7))
	MOVV	$64, R7
	BLT	R5, R7, clr_33through64
clr_64loop:
	SUBV	$64, R5
	SGTU    $64, R5, R7
	MOVV	R0, (R4)
	MOVV	R0, 8(R4)
	MOVV	R0, 16(R4)
	MOVV	R0, 24(R4)
	MOVV	R0, 32(R4)
	MOVV	R0, 40(R4)
	MOVV	R0, 48(R4)
	MOVV	R0, 56(R4)
	ADDV	$64, R4
	BEQ     R7, clr_64loop

	// remaining_length is 0
	BEQ	R5, clr_0

	// 32 < remaining_length < 64
	SGTU	$33, R5, R7
	BEQ	R7, clr_33through64

	// 16 < remaining_length <= 32
	SGTU	$17, R5, R7
	BEQ	R7, clr_17through32

	// 0 < remaining_length <= 16
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

lasx_clr_17through32:
	VMOVQ	V0, 0(R4)
	VMOVQ	V0, -16(R6)
	RET
lasx_clr_33through64:
	XVMOVQ	X0, 0(R4)
	XVMOVQ	X0, -32(R6)
	RET
lasx_clr_65through128:
	XVMOVQ	X0, 0(R4)
	XVMOVQ	X0, 32(R4)
	XVMOVQ	X0, -64(R6)
	XVMOVQ	X0, -32(R6)
	RET
lasx_clr_129through256:
	XVMOVQ	X0, 0(R4)
	XVMOVQ	X0, 32(R4)
	XVMOVQ	X0, 64(R4)
	XVMOVQ	X0, 96(R4)
	XVMOVQ	X0, -128(R6)
	XVMOVQ	X0, -96(R6)
	XVMOVQ	X0, -64(R6)
	XVMOVQ	X0, -32(R6)
	RET

lsx_clr_17through32:
	VMOVQ	V0, 0(R4)
	VMOVQ	V0, -16(R6)
	RET
lsx_clr_33through64:
	VMOVQ	V0, 0(R4)
	VMOVQ	V0, 16(R4)
	VMOVQ	V0, -32(R6)
	VMOVQ	V0, -16(R6)
	RET
lsx_clr_65through128:
	VMOVQ	V0, 0(R4)
	VMOVQ	V0, 16(R4)
	VMOVQ	V0, 32(R4)
	VMOVQ	V0, 48(R4)
	VMOVQ	V0, -64(R6)
	VMOVQ	V0, -48(R6)
	VMOVQ	V0, -32(R6)
	VMOVQ	V0, -16(R6)
	RET

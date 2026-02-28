// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "asm_riscv64.h"
#include "go_asm.h"
#include "textflag.h"

TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT,$0-40
	// X10 = b_base
	// X11 = b_len
	// X12 = b_cap (unused)
	// X13 = byte to find
	AND	$0xff, X13, X12		// x12 byte to look for

	SLTI	$24, X11, X14
	BNEZ	X14, small
	JMP	indexByteBig<>(SB)

small:
	MOV	X10, X13		// store base for later
	ADD	X10, X11		// end
	SUB	$1, X10
loop:
	ADD	$1, X10
	BEQ	X10, X11, notfound
	MOVBU	(X10), X14
	BNE	X12, X14, loop

	SUB	X13, X10		// remove base
	RET

notfound:
	MOV	$-1, X10
	RET

TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT,$0-32
	// X10 = b_base
	// X11 = b_len
	// X12 = byte to find
	AND	$0xff, X12		// x12 byte to look for

	SLTI	$24, X11, X14
	BNEZ	X14, small
	JMP	indexByteBig<>(SB)

small:
	MOV	X10, X13		// store base for later
	ADD	X10, X11		// end
	SUB	$1, X10
loop:
	ADD	$1, X10
	BEQ	X10, X11, notfound
	MOVBU	(X10), X14
	BNE	X12, X14, loop

	SUB	X13, X10		// remove base
	RET

notfound:
	MOV	$-1, X10
	RET

TEXT indexByteBig<>(SB),NOSPLIT|NOFRAME,$0
	// On entry:
	// X10 = b_base
	// X11 = b_len (at least 16 bytes)
	// X12 = byte to find
	// On exit:
	// X10 = index of first instance of sought byte, if found, or -1 otherwise

	MOV	X10, X13		// store base for later

#ifndef hasV
	MOVB	internal∕cpu·RISCV64+const_offsetRISCV64HasV(SB), X5
	BEQZ	X5, indexbyte_scalar
#endif

	PCALIGN	$16
vector_loop:
	VSETVLI	X11, E8, M8, TA, MA, X5
	VLE8V	(X10), V8
	VMSEQVX	X12, V8, V0
	VFIRSTM	V0, X6
	BGEZ	X6, vector_found
	ADD	X5, X10
	SUB	X5, X11
	BNEZ	X11, vector_loop
	JMP	notfound

vector_found:
	SUB	X13, X10
	ADD	X6, X10
	RET

indexbyte_scalar:
	ADD	X10, X11		// end

	// Process the first few bytes until we get to an 8 byte boundary
	// No need to check for end here as we have at least 16 bytes in
	// the buffer.

unalignedloop:
	AND	$7, X10, X14
	BEQZ	X14, aligned
	MOVBU	(X10), X14
	BEQ	X12, X14, found
	ADD	$1, X10
	JMP	unalignedloop

aligned:
	AND	$~7, X11, X15		// X15 = end of aligned data

	// We have at least 9 bytes left

	// Use 'Determine if a word has a byte equal to n' bit hack from
	// https://graphics.stanford.edu/~seander/bithacks.html to determine
	// whether the byte is present somewhere in the next 8 bytes of the
	// array.

	MOV	$0x0101010101010101, X16
	SLLI	$7, X16, X17		// X17 = 0x8080808080808080

	MUL	X12, X16, X18		// broadcast X12 to every byte in X18

alignedloop:
	MOV	(X10), X14
	XOR	X14, X18, X19

	// If the LSB in X12 is present somewhere in the 8 bytes we've just
	// loaded into X14 then at least one of the bytes in X19 will be 0
	// after the XOR.  If any of the bytes in X19 are zero then
	//
	// ((X19 - X16) & (~X19) & X17)
	//
	// will be non-zero.  The expression will evaluate to zero if none of
	// the bytes in X19 are zero, i.e., X12 is not present in X14.

	SUB	X16, X19, X20
	ANDN	X19, X17, X21
	AND	X20, X21
	BNEZ	X21, tailloop		// If X21 != 0 X12 is present in X14
	ADD	$8, X10
	BNE	X10, X15, alignedloop

tailloop:
	SUB	$1, X10

loop:
	ADD	$1, X10
	BEQ	X10, X11, notfound
	MOVBU	(X10), X14
	BNE	X12, X14, loop

found:
	SUB	X13, X10		// remove base
	RET

notfound:
	MOV	$-1, X10
	RET

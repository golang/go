// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func MemHash32(p unsafe.Pointer, h uintptr) uintptr
// ABIInternal for performance.
TEXT ·MemHash32<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to data
	// BX = seed
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	MOVQ	BX, X0	// X0 = seed
	PINSRD	$2, (AX), X0	// data
	AESENC	·aeskeysched+0(SB), X0
	AESENC	·aeskeysched+16(SB), X0
	AESENC	·aeskeysched+32(SB), X0
	MOVQ	X0, AX	// return X0
	RET
noaes:
	JMP	·memHash32Fallback<ABIInternal>(SB)

// func MemHash64(p unsafe.Pointer, h uintptr) uintptr
// ABIInternal for performance.
TEXT ·MemHash64<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to data
	// BX = seed
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	MOVQ	BX, X0	// X0 = seed
	PINSRQ	$1, (AX), X0	// data
	AESENC	·aeskeysched+0(SB), X0
	AESENC	·aeskeysched+16(SB), X0
	AESENC	·aeskeysched+32(SB), X0
	MOVQ	X0, AX	// return X0
	RET
noaes:
	JMP	·memHash64Fallback<ABIInternal>(SB)

// func MemHash(p unsafe.Pointer, h, s uintptr) uintptr
// hash function using AES hardware instructions
TEXT ·MemHash<ABIInternal>(SB),NOSPLIT,$0-32
	// AX = ptr to data
	// BX = seed
	// CX = size
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	JMP	·aeshashbody<>(SB)
noaes:
	JMP	·memHashFallback<ABIInternal>(SB)

// func strhash(p unsafe.Pointer, h uintptr) uintptr
TEXT ·StrHash<ABIInternal>(SB),NOSPLIT,$0-24
	// AX = ptr to string struct
	// BX = seed
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	MOVQ	8(AX), CX	// length of string
	MOVQ	(AX), AX	// string data
	JMP	·aeshashbody<>(SB)
noaes:
	JMP	·strHashFallback<ABIInternal>(SB)

// AX: data
// BX: hash seed
// CX: length
// At return: AX = return value
TEXT ·aeshashbody<>(SB),NOSPLIT,$0-0
	// Fill an SSE register with our seeds.
	MOVQ	BX, X0				// 64 bits of per-table hash seed
	PINSRW	$4, CX, X0			// 16 bits of length
	PSHUFHW $0, X0, X0			// repeat length 4 times total
	MOVO	X0, X1				// save unscrambled seed
	PXOR	·aeskeysched(SB), X0	// xor in per-process seed
	AESENC	X0, X0				// scramble seed

	CMPQ	CX, $16
	JB	aes0to15
	JE	aes16
	CMPQ	CX, $32
	JBE	aes17to32
	CMPQ	CX, $64
	JBE	aes33to64
	CMPQ	CX, $128
	JBE	aes65to128
	JMP	aes129plus

aes0to15:
	TESTQ	CX, CX
	JE	aes0

	ADDQ	$16, AX
	TESTW	$0xff0, AX
	JE	endofpage

	// 16 bytes loaded at this address won't cross
	// a page boundary, so we can load it directly.
	MOVOU	-16(AX), X1
	ADDQ	CX, CX
	MOVQ	$masks<>(SB), AX
	PAND	(AX)(CX*8), X1
final1:
	PXOR	X0, X1	// xor data with seed
	AESENC	X1, X1	// scramble combo 3 times
	AESENC	X1, X1
	AESENC	X1, X1
	MOVQ	X1, AX	// return X1
	RET

endofpage:
	// address ends in 1111xxxx. Might be up against
	// a page boundary, so load ending at last byte.
	// Then shift bytes down using pshufb.
	MOVOU	-32(AX)(CX*1), X1
	ADDQ	CX, CX
	MOVQ	$shifts<>(SB), AX
	PSHUFB	(AX)(CX*8), X1
	JMP	final1

aes0:
	// Return scrambled input seed
	AESENC	X0, X0
	MOVQ	X0, AX	// return X0
	RET

aes16:
	MOVOU	(AX), X1
	JMP	final1

aes17to32:
	// make second starting seed
	PXOR	·aeskeysched+16(SB), X1
	AESENC	X1, X1

	// load data to be hashed
	MOVOU	(AX), X2
	MOVOU	-16(AX)(CX*1), X3

	// xor with seed
	PXOR	X0, X2
	PXOR	X1, X3

	// scramble 3 times
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X2, X2
	AESENC	X3, X3

	// combine results
	PXOR	X3, X2
	MOVQ	X2, AX	// return X2
	RET

aes33to64:
	// make 3 more starting seeds
	MOVO	X1, X2
	MOVO	X1, X3
	PXOR	·aeskeysched+16(SB), X1
	PXOR	·aeskeysched+32(SB), X2
	PXOR	·aeskeysched+48(SB), X3
	AESENC	X1, X1
	AESENC	X2, X2
	AESENC	X3, X3

	MOVOU	(AX), X4
	MOVOU	16(AX), X5
	MOVOU	-32(AX)(CX*1), X6
	MOVOU	-16(AX)(CX*1), X7

	PXOR	X0, X4
	PXOR	X1, X5
	PXOR	X2, X6
	PXOR	X3, X7

	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	PXOR	X6, X4
	PXOR	X7, X5
	PXOR	X5, X4
	MOVQ	X4, AX	// return X4
	RET

aes65to128:
	// make 7 more starting seeds
	MOVO	X1, X2
	MOVO	X1, X3
	MOVO	X1, X4
	MOVO	X1, X5
	MOVO	X1, X6
	MOVO	X1, X7
	PXOR	·aeskeysched+16(SB), X1
	PXOR	·aeskeysched+32(SB), X2
	PXOR	·aeskeysched+48(SB), X3
	PXOR	·aeskeysched+64(SB), X4
	PXOR	·aeskeysched+80(SB), X5
	PXOR	·aeskeysched+96(SB), X6
	PXOR	·aeskeysched+112(SB), X7
	AESENC	X1, X1
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	// load data
	MOVOU	(AX), X8
	MOVOU	16(AX), X9
	MOVOU	32(AX), X10
	MOVOU	48(AX), X11
	MOVOU	-64(AX)(CX*1), X12
	MOVOU	-48(AX)(CX*1), X13
	MOVOU	-32(AX)(CX*1), X14
	MOVOU	-16(AX)(CX*1), X15

	// xor with seed
	PXOR	X0, X8
	PXOR	X1, X9
	PXOR	X2, X10
	PXOR	X3, X11
	PXOR	X4, X12
	PXOR	X5, X13
	PXOR	X6, X14
	PXOR	X7, X15

	// scramble 3 times
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	// combine results
	PXOR	X12, X8
	PXOR	X13, X9
	PXOR	X14, X10
	PXOR	X15, X11
	PXOR	X10, X8
	PXOR	X11, X9
	PXOR	X9, X8
	// X15 must be zero on return
	PXOR	X15, X15
	MOVQ	X8, AX	// return X8
	RET

aes129plus:
	// make 7 more starting seeds
	MOVO	X1, X2
	MOVO	X1, X3
	MOVO	X1, X4
	MOVO	X1, X5
	MOVO	X1, X6
	MOVO	X1, X7
	PXOR	·aeskeysched+16(SB), X1
	PXOR	·aeskeysched+32(SB), X2
	PXOR	·aeskeysched+48(SB), X3
	PXOR	·aeskeysched+64(SB), X4
	PXOR	·aeskeysched+80(SB), X5
	PXOR	·aeskeysched+96(SB), X6
	PXOR	·aeskeysched+112(SB), X7
	AESENC	X1, X1
	AESENC	X2, X2
	AESENC	X3, X3
	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	// start with last (possibly overlapping) block
	MOVOU	-128(AX)(CX*1), X8
	MOVOU	-112(AX)(CX*1), X9
	MOVOU	-96(AX)(CX*1), X10
	MOVOU	-80(AX)(CX*1), X11
	MOVOU	-64(AX)(CX*1), X12
	MOVOU	-48(AX)(CX*1), X13
	MOVOU	-32(AX)(CX*1), X14
	MOVOU	-16(AX)(CX*1), X15

	// xor in seed
	PXOR	X0, X8
	PXOR	X1, X9
	PXOR	X2, X10
	PXOR	X3, X11
	PXOR	X4, X12
	PXOR	X5, X13
	PXOR	X6, X14
	PXOR	X7, X15

	// compute number of remaining 128-byte blocks
	DECQ	CX
	SHRQ	$7, CX

	PCALIGN $16
aesloop:
	// scramble state
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	// scramble state, xor in a block
	MOVOU	(AX), X0
	MOVOU	16(AX), X1
	MOVOU	32(AX), X2
	MOVOU	48(AX), X3
	AESENC	X0, X8
	AESENC	X1, X9
	AESENC	X2, X10
	AESENC	X3, X11
	MOVOU	64(AX), X4
	MOVOU	80(AX), X5
	MOVOU	96(AX), X6
	MOVOU	112(AX), X7
	AESENC	X4, X12
	AESENC	X5, X13
	AESENC	X6, X14
	AESENC	X7, X15

	ADDQ	$128, AX
	DECQ	CX
	JNE	aesloop

	// 3 more scrambles to finish
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15
	AESENC	X8, X8
	AESENC	X9, X9
	AESENC	X10, X10
	AESENC	X11, X11
	AESENC	X12, X12
	AESENC	X13, X13
	AESENC	X14, X14
	AESENC	X15, X15

	PXOR	X12, X8
	PXOR	X13, X9
	PXOR	X14, X10
	PXOR	X15, X11
	PXOR	X10, X8
	PXOR	X11, X9
	PXOR	X9, X8
	// X15 must be zero on return
	PXOR	X15, X15
	MOVQ	X8, AX	// return X8
	RET

// simple mask to get rid of data in the high part of the register.
DATA masks<>+0x00(SB)/8, $0x0000000000000000
DATA masks<>+0x08(SB)/8, $0x0000000000000000
DATA masks<>+0x10(SB)/8, $0x00000000000000ff
DATA masks<>+0x18(SB)/8, $0x0000000000000000
DATA masks<>+0x20(SB)/8, $0x000000000000ffff
DATA masks<>+0x28(SB)/8, $0x0000000000000000
DATA masks<>+0x30(SB)/8, $0x0000000000ffffff
DATA masks<>+0x38(SB)/8, $0x0000000000000000
DATA masks<>+0x40(SB)/8, $0x00000000ffffffff
DATA masks<>+0x48(SB)/8, $0x0000000000000000
DATA masks<>+0x50(SB)/8, $0x000000ffffffffff
DATA masks<>+0x58(SB)/8, $0x0000000000000000
DATA masks<>+0x60(SB)/8, $0x0000ffffffffffff
DATA masks<>+0x68(SB)/8, $0x0000000000000000
DATA masks<>+0x70(SB)/8, $0x00ffffffffffffff
DATA masks<>+0x78(SB)/8, $0x0000000000000000
DATA masks<>+0x80(SB)/8, $0xffffffffffffffff
DATA masks<>+0x88(SB)/8, $0x0000000000000000
DATA masks<>+0x90(SB)/8, $0xffffffffffffffff
DATA masks<>+0x98(SB)/8, $0x00000000000000ff
DATA masks<>+0xa0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xa8(SB)/8, $0x000000000000ffff
DATA masks<>+0xb0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xb8(SB)/8, $0x0000000000ffffff
DATA masks<>+0xc0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xc8(SB)/8, $0x00000000ffffffff
DATA masks<>+0xd0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xd8(SB)/8, $0x000000ffffffffff
DATA masks<>+0xe0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xe8(SB)/8, $0x0000ffffffffffff
DATA masks<>+0xf0(SB)/8, $0xffffffffffffffff
DATA masks<>+0xf8(SB)/8, $0x00ffffffffffffff
GLOBL masks<>(SB),RODATA,$256

// these are arguments to pshufb. They move data down from
// the high bytes of the register to the low bytes of the register.
// index is how many bytes to move.
DATA shifts<>+0x00(SB)/8, $0x0000000000000000
DATA shifts<>+0x08(SB)/8, $0x0000000000000000
DATA shifts<>+0x10(SB)/8, $0xffffffffffffff0f
DATA shifts<>+0x18(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x20(SB)/8, $0xffffffffffff0f0e
DATA shifts<>+0x28(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x30(SB)/8, $0xffffffffff0f0e0d
DATA shifts<>+0x38(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x40(SB)/8, $0xffffffff0f0e0d0c
DATA shifts<>+0x48(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x50(SB)/8, $0xffffff0f0e0d0c0b
DATA shifts<>+0x58(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x60(SB)/8, $0xffff0f0e0d0c0b0a
DATA shifts<>+0x68(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x70(SB)/8, $0xff0f0e0d0c0b0a09
DATA shifts<>+0x78(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x80(SB)/8, $0x0f0e0d0c0b0a0908
DATA shifts<>+0x88(SB)/8, $0xffffffffffffffff
DATA shifts<>+0x90(SB)/8, $0x0e0d0c0b0a090807
DATA shifts<>+0x98(SB)/8, $0xffffffffffffff0f
DATA shifts<>+0xa0(SB)/8, $0x0d0c0b0a09080706
DATA shifts<>+0xa8(SB)/8, $0xffffffffffff0f0e
DATA shifts<>+0xb0(SB)/8, $0x0c0b0a0908070605
DATA shifts<>+0xb8(SB)/8, $0xffffffffff0f0e0d
DATA shifts<>+0xc0(SB)/8, $0x0b0a090807060504
DATA shifts<>+0xc8(SB)/8, $0xffffffff0f0e0d0c
DATA shifts<>+0xd0(SB)/8, $0x0a09080706050403
DATA shifts<>+0xd8(SB)/8, $0xffffff0f0e0d0c0b
DATA shifts<>+0xe0(SB)/8, $0x0908070605040302
DATA shifts<>+0xe8(SB)/8, $0xffff0f0e0d0c0b0a
DATA shifts<>+0xf0(SB)/8, $0x0807060504030201
DATA shifts<>+0xf8(SB)/8, $0xff0f0e0d0c0b0a09
GLOBL shifts<>(SB),RODATA,$256

TEXT ·checkMasksAndShiftsAlignment<ABIInternal>(SB),NOSPLIT,$0-1
	// check that masks<>(SB) and shifts<>(SB) are aligned to 16-byte
	MOVQ	$masks<>(SB), AX
	MOVQ	$shifts<>(SB), BX
	ORQ	BX, AX
	TESTQ	$15, AX
	SETEQ	AX
	RET

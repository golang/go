// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// hash function using AES hardware instructions
TEXT ·MemHash32(SB),NOSPLIT,$0-12
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	MOVL	p+0(FP), AX	// ptr to data
	MOVL	h+4(FP), X0	// seed
	PINSRD	$1, (AX), X0	// data
	AESENC	·aeskeysched+0(SB), X0
	AESENC	·aeskeysched+16(SB), X0
	AESENC	·aeskeysched+32(SB), X0
	MOVL	X0, ret+8(FP)
	RET
noaes:
	JMP	·memHash32Fallback(SB)

TEXT ·MemHash64(SB),NOSPLIT,$0-12
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	MOVL	p+0(FP), AX	// ptr to data
	MOVQ	(AX), X0	// data
	PINSRD	$2, h+4(FP), X0	// seed
	AESENC	·aeskeysched+0(SB), X0
	AESENC	·aeskeysched+16(SB), X0
	AESENC	·aeskeysched+32(SB), X0
	MOVL	X0, ret+8(FP)
	RET
noaes:
	JMP	·memHash64Fallback(SB)

TEXT ·MemHash(SB),NOSPLIT,$0-16
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	MOVL	p+0(FP), AX	// ptr to data
	MOVL	s+8(FP), BX	// size
	LEAL	ret+12(FP), DX
	JMP	·aeshashbody<>(SB)
noaes:
	JMP	·memHashFallback(SB)

TEXT ·StrHash(SB),NOSPLIT,$0-12
	CMPB	·UseAeshash(SB), $0
	JEQ	noaes
	MOVL	p+0(FP), AX	// ptr to string object
	MOVL	4(AX), BX	// length of string
	MOVL	(AX), AX	// string data
	LEAL	ret+8(FP), DX
	JMP	·aeshashbody<>(SB)
noaes:
	JMP	·strHashFallback(SB)

// AX: data
// BX: length
// DX: address to put return value
TEXT ·aeshashbody<>(SB),NOSPLIT,$0-0
	MOVL	h+4(FP), X0	            // 32 bits of per-table hash seed
	PINSRW	$4, BX, X0	            // 16 bits of length
	PSHUFHW	$0, X0, X0	            // replace size with its low 2 bytes repeated 4 times
	MOVO	X0, X1                      // save unscrambled seed
	PXOR	·aeskeysched(SB), X0 // xor in per-process seed
	AESENC	X0, X0                      // scramble seed

	CMPL	BX, $16
	JB	aes0to15
	JE	aes16
	CMPL	BX, $32
	JBE	aes17to32
	CMPL	BX, $64
	JBE	aes33to64
	JMP	aes65plus

aes0to15:
	TESTL	BX, BX
	JE	aes0

	ADDL	$16, AX
	TESTW	$0xff0, AX
	JE	endofpage

	// 16 bytes loaded at this address won't cross
	// a page boundary, so we can load it directly.
	MOVOU	-16(AX), X1
	ADDL	BX, BX
	PAND	masks<>(SB)(BX*8), X1

final1:
	PXOR	X0, X1	// xor data with seed
	AESENC	X1, X1  // scramble combo 3 times
	AESENC	X1, X1
	AESENC	X1, X1
	MOVL	X1, (DX)
	RET

endofpage:
	// address ends in 1111xxxx. Might be up against
	// a page boundary, so load ending at last byte.
	// Then shift bytes down using pshufb.
	MOVOU	-32(AX)(BX*1), X1
	ADDL	BX, BX
	PSHUFB	shifts<>(SB)(BX*8), X1
	JMP	final1

aes0:
	// Return scrambled input seed
	AESENC	X0, X0
	MOVL	X0, (DX)
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
	MOVOU	-16(AX)(BX*1), X3

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
	MOVL	X2, (DX)
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
	MOVOU	-32(AX)(BX*1), X6
	MOVOU	-16(AX)(BX*1), X7

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
	MOVL	X4, (DX)
	RET

aes65plus:
	// make 3 more starting seeds
	MOVO	X1, X2
	MOVO	X1, X3
	PXOR	·aeskeysched+16(SB), X1
	PXOR	·aeskeysched+32(SB), X2
	PXOR	·aeskeysched+48(SB), X3
	AESENC	X1, X1
	AESENC	X2, X2
	AESENC	X3, X3

	// start with last (possibly overlapping) block
	MOVOU	-64(AX)(BX*1), X4
	MOVOU	-48(AX)(BX*1), X5
	MOVOU	-32(AX)(BX*1), X6
	MOVOU	-16(AX)(BX*1), X7

	// scramble state once
	AESENC	X0, X4
	AESENC	X1, X5
	AESENC	X2, X6
	AESENC	X3, X7

	// compute number of remaining 64-byte blocks
	DECL	BX
	SHRL	$6, BX

aesloop:
	// scramble state, xor in a block
	MOVOU	(AX), X0
	MOVOU	16(AX), X1
	MOVOU	32(AX), X2
	MOVOU	48(AX), X3
	AESENC	X0, X4
	AESENC	X1, X5
	AESENC	X2, X6
	AESENC	X3, X7

	// scramble state
	AESENC	X4, X4
	AESENC	X5, X5
	AESENC	X6, X6
	AESENC	X7, X7

	ADDL	$64, AX
	DECL	BX
	JNE	aesloop

	// 3 more scrambles to finish
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
	MOVL	X4, (DX)
	RET

// simple mask to get rid of data in the high part of the register.
DATA masks<>+0x00(SB)/4, $0x00000000
DATA masks<>+0x04(SB)/4, $0x00000000
DATA masks<>+0x08(SB)/4, $0x00000000
DATA masks<>+0x0c(SB)/4, $0x00000000

DATA masks<>+0x10(SB)/4, $0x000000ff
DATA masks<>+0x14(SB)/4, $0x00000000
DATA masks<>+0x18(SB)/4, $0x00000000
DATA masks<>+0x1c(SB)/4, $0x00000000

DATA masks<>+0x20(SB)/4, $0x0000ffff
DATA masks<>+0x24(SB)/4, $0x00000000
DATA masks<>+0x28(SB)/4, $0x00000000
DATA masks<>+0x2c(SB)/4, $0x00000000

DATA masks<>+0x30(SB)/4, $0x00ffffff
DATA masks<>+0x34(SB)/4, $0x00000000
DATA masks<>+0x38(SB)/4, $0x00000000
DATA masks<>+0x3c(SB)/4, $0x00000000

DATA masks<>+0x40(SB)/4, $0xffffffff
DATA masks<>+0x44(SB)/4, $0x00000000
DATA masks<>+0x48(SB)/4, $0x00000000
DATA masks<>+0x4c(SB)/4, $0x00000000

DATA masks<>+0x50(SB)/4, $0xffffffff
DATA masks<>+0x54(SB)/4, $0x000000ff
DATA masks<>+0x58(SB)/4, $0x00000000
DATA masks<>+0x5c(SB)/4, $0x00000000

DATA masks<>+0x60(SB)/4, $0xffffffff
DATA masks<>+0x64(SB)/4, $0x0000ffff
DATA masks<>+0x68(SB)/4, $0x00000000
DATA masks<>+0x6c(SB)/4, $0x00000000

DATA masks<>+0x70(SB)/4, $0xffffffff
DATA masks<>+0x74(SB)/4, $0x00ffffff
DATA masks<>+0x78(SB)/4, $0x00000000
DATA masks<>+0x7c(SB)/4, $0x00000000

DATA masks<>+0x80(SB)/4, $0xffffffff
DATA masks<>+0x84(SB)/4, $0xffffffff
DATA masks<>+0x88(SB)/4, $0x00000000
DATA masks<>+0x8c(SB)/4, $0x00000000

DATA masks<>+0x90(SB)/4, $0xffffffff
DATA masks<>+0x94(SB)/4, $0xffffffff
DATA masks<>+0x98(SB)/4, $0x000000ff
DATA masks<>+0x9c(SB)/4, $0x00000000

DATA masks<>+0xa0(SB)/4, $0xffffffff
DATA masks<>+0xa4(SB)/4, $0xffffffff
DATA masks<>+0xa8(SB)/4, $0x0000ffff
DATA masks<>+0xac(SB)/4, $0x00000000

DATA masks<>+0xb0(SB)/4, $0xffffffff
DATA masks<>+0xb4(SB)/4, $0xffffffff
DATA masks<>+0xb8(SB)/4, $0x00ffffff
DATA masks<>+0xbc(SB)/4, $0x00000000

DATA masks<>+0xc0(SB)/4, $0xffffffff
DATA masks<>+0xc4(SB)/4, $0xffffffff
DATA masks<>+0xc8(SB)/4, $0xffffffff
DATA masks<>+0xcc(SB)/4, $0x00000000

DATA masks<>+0xd0(SB)/4, $0xffffffff
DATA masks<>+0xd4(SB)/4, $0xffffffff
DATA masks<>+0xd8(SB)/4, $0xffffffff
DATA masks<>+0xdc(SB)/4, $0x000000ff

DATA masks<>+0xe0(SB)/4, $0xffffffff
DATA masks<>+0xe4(SB)/4, $0xffffffff
DATA masks<>+0xe8(SB)/4, $0xffffffff
DATA masks<>+0xec(SB)/4, $0x0000ffff

DATA masks<>+0xf0(SB)/4, $0xffffffff
DATA masks<>+0xf4(SB)/4, $0xffffffff
DATA masks<>+0xf8(SB)/4, $0xffffffff
DATA masks<>+0xfc(SB)/4, $0x00ffffff

GLOBL masks<>(SB),RODATA,$256

// these are arguments to pshufb. They move data down from
// the high bytes of the register to the low bytes of the register.
// index is how many bytes to move.
DATA shifts<>+0x00(SB)/4, $0x00000000
DATA shifts<>+0x04(SB)/4, $0x00000000
DATA shifts<>+0x08(SB)/4, $0x00000000
DATA shifts<>+0x0c(SB)/4, $0x00000000

DATA shifts<>+0x10(SB)/4, $0xffffff0f
DATA shifts<>+0x14(SB)/4, $0xffffffff
DATA shifts<>+0x18(SB)/4, $0xffffffff
DATA shifts<>+0x1c(SB)/4, $0xffffffff

DATA shifts<>+0x20(SB)/4, $0xffff0f0e
DATA shifts<>+0x24(SB)/4, $0xffffffff
DATA shifts<>+0x28(SB)/4, $0xffffffff
DATA shifts<>+0x2c(SB)/4, $0xffffffff

DATA shifts<>+0x30(SB)/4, $0xff0f0e0d
DATA shifts<>+0x34(SB)/4, $0xffffffff
DATA shifts<>+0x38(SB)/4, $0xffffffff
DATA shifts<>+0x3c(SB)/4, $0xffffffff

DATA shifts<>+0x40(SB)/4, $0x0f0e0d0c
DATA shifts<>+0x44(SB)/4, $0xffffffff
DATA shifts<>+0x48(SB)/4, $0xffffffff
DATA shifts<>+0x4c(SB)/4, $0xffffffff

DATA shifts<>+0x50(SB)/4, $0x0e0d0c0b
DATA shifts<>+0x54(SB)/4, $0xffffff0f
DATA shifts<>+0x58(SB)/4, $0xffffffff
DATA shifts<>+0x5c(SB)/4, $0xffffffff

DATA shifts<>+0x60(SB)/4, $0x0d0c0b0a
DATA shifts<>+0x64(SB)/4, $0xffff0f0e
DATA shifts<>+0x68(SB)/4, $0xffffffff
DATA shifts<>+0x6c(SB)/4, $0xffffffff

DATA shifts<>+0x70(SB)/4, $0x0c0b0a09
DATA shifts<>+0x74(SB)/4, $0xff0f0e0d
DATA shifts<>+0x78(SB)/4, $0xffffffff
DATA shifts<>+0x7c(SB)/4, $0xffffffff

DATA shifts<>+0x80(SB)/4, $0x0b0a0908
DATA shifts<>+0x84(SB)/4, $0x0f0e0d0c
DATA shifts<>+0x88(SB)/4, $0xffffffff
DATA shifts<>+0x8c(SB)/4, $0xffffffff

DATA shifts<>+0x90(SB)/4, $0x0a090807
DATA shifts<>+0x94(SB)/4, $0x0e0d0c0b
DATA shifts<>+0x98(SB)/4, $0xffffff0f
DATA shifts<>+0x9c(SB)/4, $0xffffffff

DATA shifts<>+0xa0(SB)/4, $0x09080706
DATA shifts<>+0xa4(SB)/4, $0x0d0c0b0a
DATA shifts<>+0xa8(SB)/4, $0xffff0f0e
DATA shifts<>+0xac(SB)/4, $0xffffffff

DATA shifts<>+0xb0(SB)/4, $0x08070605
DATA shifts<>+0xb4(SB)/4, $0x0c0b0a09
DATA shifts<>+0xb8(SB)/4, $0xff0f0e0d
DATA shifts<>+0xbc(SB)/4, $0xffffffff

DATA shifts<>+0xc0(SB)/4, $0x07060504
DATA shifts<>+0xc4(SB)/4, $0x0b0a0908
DATA shifts<>+0xc8(SB)/4, $0x0f0e0d0c
DATA shifts<>+0xcc(SB)/4, $0xffffffff

DATA shifts<>+0xd0(SB)/4, $0x06050403
DATA shifts<>+0xd4(SB)/4, $0x0a090807
DATA shifts<>+0xd8(SB)/4, $0x0e0d0c0b
DATA shifts<>+0xdc(SB)/4, $0xffffff0f

DATA shifts<>+0xe0(SB)/4, $0x05040302
DATA shifts<>+0xe4(SB)/4, $0x09080706
DATA shifts<>+0xe8(SB)/4, $0x0d0c0b0a
DATA shifts<>+0xec(SB)/4, $0xffff0f0e

DATA shifts<>+0xf0(SB)/4, $0x04030201
DATA shifts<>+0xf4(SB)/4, $0x08070605
DATA shifts<>+0xf8(SB)/4, $0x0c0b0a09
DATA shifts<>+0xfc(SB)/4, $0xff0f0e0d

GLOBL shifts<>(SB),RODATA,$256

TEXT ·checkMasksAndShiftsAlignment(SB),NOSPLIT,$0-1
	// check that masks<>(SB) and shifts<>(SB) are aligned to 16-byte
	MOVL	$masks<>(SB), AX
	MOVL	$shifts<>(SB), BX
	ORL	BX, AX
	TESTL	$15, AX
	SETEQ   ret+0(FP)
	RET

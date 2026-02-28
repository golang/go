// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Based on CRYPTOGAMS code with the following comment:
// # ====================================================================
// # Written by Andy Polyakov <appro@openssl.org> for the OpenSSL
// # project. The module is, however, dual licensed under OpenSSL and
// # CRYPTOGAMS licenses depending on where you obtain it. For further
// # details see http://www.openssl.org/~appro/cryptogams/.
// # ====================================================================

// Original code can be found at the link below:
// https://github.com/dot-asm/cryptogams/blob/master/ppc/aesp8-ppc.pl

// Some function names were changed to be consistent with Go function
// names. For instance, function aes_p8_set_{en,de}crypt_key become
// set{En,De}cryptKeyAsm. I also split setEncryptKeyAsm in two parts
// and a new session was created (doEncryptKeyAsm). This was necessary to
// avoid arguments overwriting when setDecryptKeyAsm calls setEncryptKeyAsm.
// There were other modifications as well but kept the same functionality.

#include "textflag.h"

// For expandKeyAsm
#define INP     R3
#define BITS    R4
#define OUTENC  R5 // Pointer to next expanded encrypt key
#define PTR     R6
#define CNT     R7
#define ROUNDS  R8
#define OUTDEC  R9  // Pointer to next expanded decrypt key
#define TEMP    R19
#define ZERO    V0
#define IN0     V1
#define IN1     V2
#define KEY     V3
#define RCON    V4
#define MASK    V5
#define TMP     V6
#define STAGE   V7
#define OUTPERM V8
#define OUTMASK V9
#define OUTHEAD V10
#define OUTTAIL V11

// For P9 instruction emulation
#define ESPERM  V21  // Endian swapping permute into BE
#define TMP2    V22  // Temporary for P8_STXVB16X/P8_STXV

// For {en,de}cryptBlockAsm
#define BLK_INP    R3
#define BLK_OUT    R4
#define BLK_KEY    R5
#define BLK_ROUNDS R6
#define BLK_IDX    R7

DATA ·rcon+0x00(SB)/8, $0x0f0e0d0c0b0a0908 // Permute for vector doubleword endian swap
DATA ·rcon+0x08(SB)/8, $0x0706050403020100
DATA ·rcon+0x10(SB)/8, $0x0100000001000000 // RCON
DATA ·rcon+0x18(SB)/8, $0x0100000001000000 // RCON
DATA ·rcon+0x20(SB)/8, $0x1b0000001b000000
DATA ·rcon+0x28(SB)/8, $0x1b0000001b000000
DATA ·rcon+0x30(SB)/8, $0x0d0e0f0c0d0e0f0c // MASK
DATA ·rcon+0x38(SB)/8, $0x0d0e0f0c0d0e0f0c // MASK
DATA ·rcon+0x40(SB)/8, $0x0000000000000000
DATA ·rcon+0x48(SB)/8, $0x0000000000000000
GLOBL ·rcon(SB), RODATA, $80

// Emulate unaligned BE vector load/stores on LE targets
#define P8_LXVB16X(RA,RB,VT) \
	LXVD2X	(RA+RB), VT \
	VPERM	VT, VT, ESPERM, VT

#define P8_STXVB16X(VS,RA,RB) \
	VPERM	VS, VS, ESPERM, TMP2 \
	STXVD2X	TMP2, (RA+RB)

#define P8_STXV(VS,RA,RB) \
	XXPERMDI	VS, VS, $2, TMP2 \
	STXVD2X		TMP2, (RA+RB)

#define P8_LXV(RA,RB,VT) \
	LXVD2X		(RA+RB), VT \
	XXPERMDI	VT, VT, $2, VT

#define LXSDX_BE(RA,RB,VT) \
	LXSDX	(RA+RB), VT \
	VPERM	VT, VT, ESPERM, VT

// func setEncryptKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)
TEXT ·expandKeyAsm(SB), NOSPLIT|NOFRAME, $0
	// Load the arguments inside the registers
	MOVD	nr+0(FP), ROUNDS
	MOVD	key+8(FP), INP
	MOVD	enc+16(FP), OUTENC
	MOVD	dec+24(FP), OUTDEC

	MOVD	$·rcon(SB), PTR // PTR point to rcon addr
	LVX	(PTR), ESPERM
	ADD	$0x10, PTR

	// Get key from memory and write aligned into VR
	P8_LXVB16X(INP, R0, IN0)
	ADD	$0x10, INP, INP
	MOVD	$0x20, TEMP

	CMPW	ROUNDS, $12
	LVX	(PTR)(R0), RCON    // lvx   4,0,6      Load first 16 bytes into RCON
	LVX	(PTR)(TEMP), MASK
	ADD	$0x10, PTR, PTR    // addi  6,6,0x10   PTR to next 16 bytes of RCON
	MOVD	$8, CNT            // li    7,8        CNT = 8
	VXOR	ZERO, ZERO, ZERO   // vxor  0,0,0      Zero to be zero :)
	MOVD	CNT, CTR           // mtctr 7          Set the counter to 8 (rounds)

	// The expanded decrypt key is the expanded encrypt key stored in reverse order.
	// Move OUTDEC to the last key location, and store in descending order.
	ADD	$160, OUTDEC, OUTDEC
	BLT	loop128
	ADD	$32, OUTDEC, OUTDEC
	BEQ	l192
	ADD	$32, OUTDEC, OUTDEC
	JMP	l256

loop128:
	// Key schedule (Round 1 to 8)
	VPERM	IN0, IN0, MASK, KEY              // vperm 3,1,1,5         Rotate-n-splat
	VSLDOI	$12, ZERO, IN0, TMP              // vsldoi 6,0,1,12
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)
	VCIPHERLAST	KEY, RCON, KEY           // vcipherlast 3,3,4
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC

	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VADDUWM	RCON, RCON, RCON    // vadduwm 4,4,4
	VXOR	IN0, KEY, IN0       // vxor 1,1,3
	BC	0x10, 0, loop128    // bdnz .Loop128

	LVX	(PTR)(R0), RCON // lvx 4,0,6     Last two round keys

	// Key schedule (Round 9)
	VPERM	IN0, IN0, MASK, KEY              // vperm 3,1,1,5   Rotate-n-spat
	VSLDOI	$12, ZERO, IN0, TMP              // vsldoi 6,0,1,12
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)
	VCIPHERLAST	KEY, RCON, KEY           // vcipherlast 3,3,4
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC

	// Key schedule (Round 10)
	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VADDUWM	RCON, RCON, RCON    // vadduwm 4,4,4
	VXOR	IN0, KEY, IN0       // vxor 1,1,3

	VPERM	IN0, IN0, MASK, KEY              // vperm 3,1,1,5   Rotate-n-splat
	VSLDOI	$12, ZERO, IN0, TMP              // vsldoi 6,0,1,12
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)
	VCIPHERLAST	KEY, RCON, KEY           // vcipherlast 3,3,4
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC

	// Key schedule (Round 11)
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP              // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP              // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VXOR	IN0, KEY, IN0                    // vxor 1,1,3
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)

	RET

l192:
	LXSDX_BE(INP, R0, IN1)                   // Load next 8 bytes into upper half of VSR in BE order.
	MOVD	$4, CNT                          // li 7,4
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC
	VSPLTISB	$8, KEY                  // vspltisb 3,8
	MOVD	CNT, CTR                         // mtctr 7
	VSUBUBM	MASK, KEY, MASK                  // vsububm 5,5,3

loop192:
	VPERM	IN1, IN1, MASK, KEY // vperm 3,2,2,5
	VSLDOI	$12, ZERO, IN0, TMP // vsldoi 6,0,1,12
	VCIPHERLAST	KEY, RCON, KEY      // vcipherlast 3,3,4

	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0       // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0       // vxor 1,1,6

	VSLDOI	$8, ZERO, IN1, STAGE  // vsldoi 7,0,2,8
	VSPLTW	$3, IN0, TMP          // vspltw 6,1,3
	VXOR	TMP, IN1, TMP         // vxor 6,6,2
	VSLDOI	$12, ZERO, IN1, IN1   // vsldoi 2,0,2,12
	VADDUWM	RCON, RCON, RCON      // vadduwm 4,4,4
	VXOR	IN1, TMP, IN1         // vxor 2,2,6
	VXOR	IN0, KEY, IN0         // vxor 1,1,3
	VXOR	IN1, KEY, IN1         // vxor 2,2,3
	VSLDOI	$8, STAGE, IN0, STAGE // vsldoi 7,7,1,8

	VPERM	IN1, IN1, MASK, KEY              // vperm 3,2,2,5
	VSLDOI	$12, ZERO, IN0, TMP              // vsldoi 6,0,1,12
	P8_STXV(STAGE, R0, OUTENC)
	P8_STXV(STAGE, R0, OUTDEC)
	VCIPHERLAST	KEY, RCON, KEY           // vcipherlast 3,3,4
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC

	VSLDOI	$8, IN0, IN1, STAGE              // vsldoi 7,1,2,8
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP              // vsldoi 6,0,6,12
	P8_STXV(STAGE, R0, OUTENC)
	P8_STXV(STAGE, R0, OUTDEC)
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP              // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC

	VSPLTW	$3, IN0, TMP                     // vspltw 6,1,3
	VXOR	TMP, IN1, TMP                    // vxor 6,6,2
	VSLDOI	$12, ZERO, IN1, IN1              // vsldoi 2,0,2,12
	VADDUWM	RCON, RCON, RCON                 // vadduwm 4,4,4
	VXOR	IN1, TMP, IN1                    // vxor 2,2,6
	VXOR	IN0, KEY, IN0                    // vxor 1,1,3
	VXOR	IN1, KEY, IN1                    // vxor 2,2,3
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC
	BC	0x10, 0, loop192                 // bdnz .Loop192

	RET

l256:
	P8_LXVB16X(INP, R0, IN1)
	MOVD	$7, CNT                          // li 7,7
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC
	MOVD	CNT, CTR                         // mtctr 7

loop256:
	VPERM	IN1, IN1, MASK, KEY              // vperm 3,2,2,5
	VSLDOI	$12, ZERO, IN0, TMP              // vsldoi 6,0,1,12
	P8_STXV(IN1, R0, OUTENC)
	P8_STXV(IN1, R0, OUTDEC)
	VCIPHERLAST	KEY, RCON, KEY           // vcipherlast 3,3,4
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC

	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP              // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP              // vsldoi 6,0,6,12
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VADDUWM	RCON, RCON, RCON                 // vadduwm 4,4,4
	VXOR	IN0, KEY, IN0                    // vxor 1,1,3
	P8_STXV(IN0, R0, OUTENC)
	P8_STXV(IN0, R0, OUTDEC)
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC
	BC	0x12, 0, done                    // bdz .Ldone

	VSPLTW	$3, IN0, KEY        // vspltw 3,1,3
	VSLDOI	$12, ZERO, IN1, TMP // vsldoi 6,0,2,12
	VSBOX	KEY, KEY            // vsbox 3,3

	VXOR	IN1, TMP, IN1       // vxor 2,2,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN1, TMP, IN1       // vxor 2,2,6
	VSLDOI	$12, ZERO, TMP, TMP // vsldoi 6,0,6,12
	VXOR	IN1, TMP, IN1       // vxor 2,2,6

	VXOR	IN1, KEY, IN1 // vxor 2,2,3
	JMP	loop256       // b .Loop256

done:
	RET

// func encryptBlockAsm(nr int, xk *uint32, dst, src *byte)
TEXT ·encryptBlockAsm(SB), NOSPLIT|NOFRAME, $0
	// Load the arguments inside the registers
	MOVD	nr+0(FP), BLK_ROUNDS
	MOVD	xk+8(FP), BLK_KEY
	MOVD	dst+16(FP), BLK_OUT
	MOVD	src+24(FP), BLK_INP

	MOVD	$15, BLK_IDX             // li 7,15

	LVX	(BLK_INP)(R0), ZERO        // lvx 0,0,3
	NEG	BLK_OUT, R11               // neg 11,4
	LVX	(BLK_INP)(BLK_IDX), IN0    // lvx 1,7,3
	LVSL	(BLK_INP)(R0), IN1         // lvsl 2,0,3
	VSPLTISB	$0x0f, RCON        // vspltisb 4,0x0f
	LVSR	(R11)(R0), KEY             // lvsr 3,0,11
	VXOR	IN1, RCON, IN1             // vxor 2,2,4
	MOVD	$16, BLK_IDX               // li 7,16
	VPERM	ZERO, IN0, IN1, ZERO       // vperm 0,0,1,2
	LVX	(BLK_KEY)(R0), IN0         // lvx 1,0,5
	LVSR	(BLK_KEY)(R0), MASK        // lvsr 5,0,5
	SRW	$1, BLK_ROUNDS, BLK_ROUNDS // srwi 6,6,1
	LVX	(BLK_KEY)(BLK_IDX), IN1    // lvx 2,7,5
	ADD	$16, BLK_IDX, BLK_IDX      // addi 7,7,16
	SUB	$1, BLK_ROUNDS, BLK_ROUNDS // subi 6,6,1
	VPERM	IN1, IN0, MASK, IN0        // vperm 1,2,1,5

	VXOR	ZERO, IN0, ZERO         // vxor 0,0,1
	LVX	(BLK_KEY)(BLK_IDX), IN0 // lvx 1,7,5
	ADD	$16, BLK_IDX, BLK_IDX   // addi 7,7,16
	MOVD	BLK_ROUNDS, CTR         // mtctr 6

loop_enc:
	VPERM	IN0, IN1, MASK, IN1     // vperm 2,1,2,5
	VCIPHER	ZERO, IN1, ZERO         // vcipher 0,0,2
	LVX	(BLK_KEY)(BLK_IDX), IN1 // lvx 2,7,5
	ADD	$16, BLK_IDX, BLK_IDX   // addi 7,7,16
	VPERM	IN1, IN0, MASK, IN0     // vperm 1,2,1,5
	VCIPHER	ZERO, IN0, ZERO         // vcipher 0,0,1
	LVX	(BLK_KEY)(BLK_IDX), IN0 // lvx 1,7,5
	ADD	$16, BLK_IDX, BLK_IDX   // addi 7,7,16
	BC	0x10, 0, loop_enc       // bdnz .Loop_enc

	VPERM	IN0, IN1, MASK, IN1     // vperm 2,1,2,5
	VCIPHER	ZERO, IN1, ZERO         // vcipher 0,0,2
	LVX	(BLK_KEY)(BLK_IDX), IN1 // lvx 2,7,5
	VPERM	IN1, IN0, MASK, IN0     // vperm 1,2,1,5
	VCIPHERLAST	ZERO, IN0, ZERO // vcipherlast 0,0,1

	VSPLTISB	$-1, IN1         // vspltisb 2,-1
	VXOR	IN0, IN0, IN0            // vxor 1,1,1
	MOVD	$15, BLK_IDX             // li 7,15
	VPERM	IN1, IN0, KEY, IN1       // vperm 2,2,1,3
	VXOR	KEY, RCON, KEY           // vxor 3,3,4
	LVX	(BLK_OUT)(R0), IN0       // lvx 1,0,4
	VPERM	ZERO, ZERO, KEY, ZERO    // vperm 0,0,0,3
	VSEL	IN0, ZERO, IN1, IN0      // vsel 1,1,0,2
	LVX	(BLK_OUT)(BLK_IDX), RCON // lvx 4,7,4
	STVX	IN0, (BLK_OUT+R0)        // stvx 1,0,4
	VSEL	ZERO, RCON, IN1, ZERO    // vsel 0,0,4,2
	STVX	ZERO, (BLK_OUT+BLK_IDX)  // stvx 0,7,4

	RET // blr

// func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)
TEXT ·decryptBlockAsm(SB), NOSPLIT|NOFRAME, $0
	// Load the arguments inside the registers
	MOVD	nr+0(FP), BLK_ROUNDS
	MOVD	xk+8(FP), BLK_KEY
	MOVD	dst+16(FP), BLK_OUT
	MOVD	src+24(FP), BLK_INP

	MOVD	$15, BLK_IDX             // li 7,15

	LVX	(BLK_INP)(R0), ZERO        // lvx 0,0,3
	NEG	BLK_OUT, R11               // neg 11,4
	LVX	(BLK_INP)(BLK_IDX), IN0    // lvx 1,7,3
	LVSL	(BLK_INP)(R0), IN1         // lvsl 2,0,3
	VSPLTISB	$0x0f, RCON        // vspltisb 4,0x0f
	LVSR	(R11)(R0), KEY             // lvsr 3,0,11
	VXOR	IN1, RCON, IN1             // vxor 2,2,4
	MOVD	$16, BLK_IDX               // li 7,16
	VPERM	ZERO, IN0, IN1, ZERO       // vperm 0,0,1,2
	LVX	(BLK_KEY)(R0), IN0         // lvx 1,0,5
	LVSR	(BLK_KEY)(R0), MASK        // lvsr 5,0,5
	SRW	$1, BLK_ROUNDS, BLK_ROUNDS // srwi 6,6,1
	LVX	(BLK_KEY)(BLK_IDX), IN1    // lvx 2,7,5
	ADD	$16, BLK_IDX, BLK_IDX      // addi 7,7,16
	SUB	$1, BLK_ROUNDS, BLK_ROUNDS // subi 6,6,1
	VPERM	IN1, IN0, MASK, IN0        // vperm 1,2,1,5

	VXOR	ZERO, IN0, ZERO         // vxor 0,0,1
	LVX	(BLK_KEY)(BLK_IDX), IN0 // lvx 1,7,5
	ADD	$16, BLK_IDX, BLK_IDX   // addi 7,7,16
	MOVD	BLK_ROUNDS, CTR         // mtctr 6

loop_dec:
	VPERM	IN0, IN1, MASK, IN1     // vperm 2,1,2,5
	VNCIPHER	ZERO, IN1, ZERO // vncipher 0,0,2
	LVX	(BLK_KEY)(BLK_IDX), IN1 // lvx 2,7,5
	ADD	$16, BLK_IDX, BLK_IDX   // addi 7,7,16
	VPERM	IN1, IN0, MASK, IN0     // vperm 1,2,1,5
	VNCIPHER	ZERO, IN0, ZERO // vncipher 0,0,1
	LVX	(BLK_KEY)(BLK_IDX), IN0 // lvx 1,7,5
	ADD	$16, BLK_IDX, BLK_IDX   // addi 7,7,16
	BC	0x10, 0, loop_dec       // bdnz .Loop_dec

	VPERM	IN0, IN1, MASK, IN1     // vperm 2,1,2,5
	VNCIPHER	ZERO, IN1, ZERO // vncipher 0,0,2
	LVX	(BLK_KEY)(BLK_IDX), IN1 // lvx 2,7,5
	VPERM	IN1, IN0, MASK, IN0     // vperm 1,2,1,5
	VNCIPHERLAST	ZERO, IN0, ZERO // vncipherlast 0,0,1

	VSPLTISB	$-1, IN1         // vspltisb 2,-1
	VXOR	IN0, IN0, IN0            // vxor 1,1,1
	MOVD	$15, BLK_IDX             // li 7,15
	VPERM	IN1, IN0, KEY, IN1       // vperm 2,2,1,3
	VXOR	KEY, RCON, KEY           // vxor 3,3,4
	LVX	(BLK_OUT)(R0), IN0       // lvx 1,0,4
	VPERM	ZERO, ZERO, KEY, ZERO    // vperm 0,0,0,3
	VSEL	IN0, ZERO, IN1, IN0      // vsel 1,1,0,2
	LVX	(BLK_OUT)(BLK_IDX), RCON // lvx 4,7,4
	STVX	IN0, (BLK_OUT+R0)        // stvx 1,0,4
	VSEL	ZERO, RCON, IN1, ZERO    // vsel 0,0,4,2
	STVX	ZERO, (BLK_OUT+BLK_IDX)  // stvx 0,7,4

	RET // blr

// Remove defines from above so they can be defined here
#undef INP
#undef OUTENC
#undef ROUNDS
#undef KEY
#undef TMP
#undef OUTPERM
#undef OUTMASK
#undef OUTHEAD
#undef OUTTAIL

// CBC encrypt or decrypt
// R3 src
// R4 dst
// R5 len
// R6 key
// R7 iv
// R8 enc=1 dec=0
// Ported from: aes_p8_cbc_encrypt
// Register usage:
// R9: ROUNDS
// R10: Index
// V0: initialized to 0
// V3: initialized to mask
// V4: IV
// V5: SRC
// V6: IV perm mask
// V7: DST
// V10: KEY perm mask

#define INP R3
#define OUT R4
#define LEN R5
#define KEY R6
#define IVP R7
#define ENC R8
#define ROUNDS R9
#define IDX R10

#define RNDKEY0 V0
#define RNDKEY1 V1
#define INOUT V2
#define TMP V3

#define IVEC V4
#define INPTAIL V5
#define INPPERM V6
#define OUTHEAD V7
#define OUTPERM V8
#define OUTMASK V9
#define KEYPERM V10

// Vector loads are done using LVX followed by
// a VPERM using mask generated from previous
// LVSL or LVSR instruction, to obtain the correct
// bytes if address is unaligned.

// Encryption is done with VCIPHER and VCIPHERLAST
// Decryption is done with VNCIPHER and VNCIPHERLAST

// Encrypt and decypt is done as follows:
// - INOUT value is initialized in outer loop.
// - ROUNDS value is adjusted for loop unrolling.
// - Encryption/decryption is done in loop based on
// adjusted ROUNDS value.
// - Final INOUT value is encrypted/decrypted and stored.

// Note: original implementation had an 8X version
// for decryption which was omitted to avoid the
// complexity.

// func cryptBlocksChain(src, dst *byte, length int, key *uint32, iv *byte, enc int, nr int)
TEXT ·cryptBlocksChain(SB), NOSPLIT|NOFRAME, $0
	MOVD	src+0(FP), INP
	MOVD	dst+8(FP), OUT
	MOVD	length+16(FP), LEN
	MOVD	key+24(FP), KEY
	MOVD	iv+32(FP), IVP
	MOVD	enc+40(FP), ENC
	MOVD	nr+48(FP), ROUNDS

	CMPU	LEN, $16                  // cmpldi r5,16
	BC	14, 0, LR                 // bltlr-
	CMPW	ENC, $0                   // cmpwi r8,0
	MOVD	$15, IDX                  // li r10,15
	VXOR	RNDKEY0, RNDKEY0, RNDKEY0 // vxor v0,v0,v0
	VSPLTISB	$0xf, TMP         // vspltisb $0xf,v3

	LVX	(IVP)(R0), IVEC                    // lvx v4,r0,r7
	LVSL	(IVP)(R0), INPPERM                 // lvsl v6,r0,r7
	LVX	(IVP)(IDX), INPTAIL                // lvx v5,r10,r7
	VXOR	INPPERM, TMP, INPPERM              // vxor v3, v6, v6
	VPERM	IVEC, INPTAIL, INPPERM, IVEC       // vperm v4,v4,v5,v6
	NEG	INP, R11                           // neg r11,r3
	LVSR	(KEY)(R0), KEYPERM                 // lvsr v10,r0,r6
	LVSR	(R11)(R0), V6                      // lvsr v6,r0,r11
	LVX	(INP)(R0), INPTAIL                 // lvx v5,r0,r3
	ADD	$15, INP                           // addi r3,r3,15
	VXOR	INPPERM, TMP, INPPERM              // vxor v6, v3, v6
	LVSL	(OUT)(R0), OUTPERM                 // lvsl v8,r0,r4
	VSPLTISB	$-1, OUTMASK               // vspltisb v9,-1
	LVX	(OUT)(R0), OUTHEAD                 // lvx v7,r0,r4
	VPERM	OUTMASK, RNDKEY0, OUTPERM, OUTMASK // vperm v9,v9,v0,v8
	VXOR	OUTPERM, TMP, OUTPERM              // vxor v8, v3, v8
	SRW	$1, ROUNDS                         // rlwinm r9,r9,31,1,31

	MOVD	$16, IDX    // li r10,16
	ADD	$-1, ROUNDS // addi r9,r9,-1
	BEQ	Lcbc_dec    // beq
	PCALIGN	$16

	// Outer loop: initialize encrypted value (INOUT)
	// Load input (INPTAIL) ivec (IVEC)
Lcbc_enc:
	VOR	INPTAIL, INPTAIL, INOUT            // vor v2,v5,v5
	LVX	(INP)(R0), INPTAIL                 // lvx v5,r0,r3
	ADD	$16, INP                           // addi r3,r3,16
	MOVD	ROUNDS, CTR                        // mtctr r9
	ADD	$-16, LEN                          // addi r5,r5,-16
	LVX	(KEY)(R0), RNDKEY0                 // lvx v0,r0,r6
	VPERM	INOUT, INPTAIL, INPPERM, INOUT     // vperm v2,v2,v5,v6
	LVX	(KEY)(IDX), RNDKEY1                // lvx v1,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	VPERM	RNDKEY1, RNDKEY0, KEYPERM, RNDKEY0 // vperm v0,v1,v0,v10
	VXOR	INOUT, RNDKEY0, INOUT              // vxor v2,v2,v0
	LVX	(KEY)(IDX), RNDKEY0                // lvx v0,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	VXOR	INOUT, IVEC, INOUT                 // vxor v2,v2,v4

	// Encryption loop of INOUT using RNDKEY0 and RNDKEY1
Loop_cbc_enc:
	VPERM	RNDKEY0, RNDKEY1, KEYPERM, RNDKEY1 // vperm v1,v1,v0,v10
	VCIPHER	INOUT, RNDKEY1, INOUT              // vcipher v2,v2,v1
	LVX	(KEY)(IDX), RNDKEY1                // lvx v1,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	VPERM	RNDKEY1, RNDKEY0, KEYPERM, RNDKEY0 // vperm v0,v0,v1,v10
	VCIPHER	INOUT, RNDKEY0, INOUT              // vcipher v2,v2,v0
	LVX	(KEY)(IDX), RNDKEY0                // lvx v0,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	BC	16, 0, Loop_cbc_enc                // bdnz Loop_cbc_enc

	// Encrypt tail values and store INOUT
	VPERM	RNDKEY0, RNDKEY1, KEYPERM, RNDKEY1 // vperm v1,v1,v0,v10
	VCIPHER	INOUT, RNDKEY1, INOUT              // vcipher v2,v2,v1
	LVX	(KEY)(IDX), RNDKEY1                // lvx v1,r10,r6
	MOVD	$16, IDX                           // li r10,16
	VPERM	RNDKEY1, RNDKEY0, KEYPERM, RNDKEY0 // vperm v0,v0,v1,v10
	VCIPHERLAST	INOUT, RNDKEY0, IVEC       // vcipherlast v4,v2,v0
	CMPU	LEN, $16                           // cmpldi r5,16
	VPERM	IVEC, IVEC, OUTPERM, TMP           // vperm v3,v4,v4,v8
	VSEL	OUTHEAD, TMP, OUTMASK, INOUT       // vsel v2,v7,v3,v9
	VOR	TMP, TMP, OUTHEAD                  // vor v7,v3,v3
	STVX	INOUT, (OUT)(R0)                   // stvx v2,r0,r4
	ADD	$16, OUT                           // addi r4,r4,16
	BGE	Lcbc_enc                           // bge Lcbc_enc
	BR	Lcbc_done                          // b Lcbc_done

	// Outer loop: initialize decrypted value (INOUT)
	// Load input (INPTAIL) ivec (IVEC)
Lcbc_dec:
	VOR	INPTAIL, INPTAIL, TMP              // vor v3,v5,v5
	LVX	(INP)(R0), INPTAIL                 // lvx v5,r0,r3
	ADD	$16, INP                           // addi r3,r3,16
	MOVD	ROUNDS, CTR                        // mtctr r9
	ADD	$-16, LEN                          // addi r5,r5,-16
	LVX	(KEY)(R0), RNDKEY0                 // lvx v0,r0,r6
	VPERM	TMP, INPTAIL, INPPERM, TMP         // vperm v3,v3,v5,v6
	LVX	(KEY)(IDX), RNDKEY1                // lvx v1,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	VPERM	RNDKEY1, RNDKEY0, KEYPERM, RNDKEY0 // vperm v0,v1,v0,v10
	VXOR	TMP, RNDKEY0, INOUT                // vxor v2,v3,v0
	LVX	(KEY)(IDX), RNDKEY0                // lvx v0,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	PCALIGN	$16

	// Decryption loop of INOUT using RNDKEY0 and RNDKEY1
Loop_cbc_dec:
	VPERM	RNDKEY0, RNDKEY1, KEYPERM, RNDKEY1 // vperm v1,v0,v1,v10
	VNCIPHER	INOUT, RNDKEY1, INOUT      // vncipher v2,v2,v1
	LVX	(KEY)(IDX), RNDKEY1                // lvx v1,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	VPERM	RNDKEY1, RNDKEY0, KEYPERM, RNDKEY0 // vperm v0,v1,v0,v10
	VNCIPHER	INOUT, RNDKEY0, INOUT      // vncipher v2,v2,v0
	LVX	(KEY)(IDX), RNDKEY0                // lvx v0,r10,r6
	ADD	$16, IDX                           // addi r10,r10,16
	BC	16, 0, Loop_cbc_dec                // bdnz

	// Decrypt tail values and store INOUT
	VPERM	RNDKEY0, RNDKEY1, KEYPERM, RNDKEY1 // vperm v1,v0,v1,v10
	VNCIPHER	INOUT, RNDKEY1, INOUT      // vncipher v2,v2,v1
	LVX	(KEY)(IDX), RNDKEY1                // lvx v1,r10,r6
	MOVD	$16, IDX                           // li r10,16
	VPERM	RNDKEY1, RNDKEY0, KEYPERM, RNDKEY0 // vperm v0,v1,v0,v10
	VNCIPHERLAST	INOUT, RNDKEY0, INOUT      // vncipherlast v2,v2,v0
	CMPU	LEN, $16                           // cmpldi r5,16
	VXOR	INOUT, IVEC, INOUT                 // vxor v2,v2,v4
	VOR	TMP, TMP, IVEC                     // vor v4,v3,v3
	VPERM	INOUT, INOUT, OUTPERM, TMP         // vperm v3,v2,v2,v8
	VSEL	OUTHEAD, TMP, OUTMASK, INOUT       // vsel v2,v7,v3,v9
	VOR	TMP, TMP, OUTHEAD                  // vor v7,v3,v3
	STVX	INOUT, (OUT)(R0)                   // stvx v2,r0,r4
	ADD	$16, OUT                           // addi r4,r4,16
	BGE	Lcbc_dec                           // bge

Lcbc_done:
	ADD	$-1, OUT                           // addi r4,r4,-1
	LVX	(OUT)(R0), INOUT                   // lvx v2,r0,r4
	VSEL	OUTHEAD, INOUT, OUTMASK, INOUT     // vsel v2,v7,v2,v9
	STVX	INOUT, (OUT)(R0)                   // stvx v2,r0,r4
	NEG	IVP, ENC                           // neg r8,r7
	MOVD	$15, IDX                           // li r10,15
	VXOR	RNDKEY0, RNDKEY0, RNDKEY0          // vxor v0,v0,v0
	VSPLTISB	$-1, OUTMASK               // vspltisb v9,-1
	VSPLTISB	$0xf, TMP                  // vspltisb v3, 0xf
	LVSR	(ENC)(R0), OUTPERM                 // lvsl v8,r0,r8
	VPERM	OUTMASK, RNDKEY0, OUTPERM, OUTMASK // vperm v9,v9,v0,v8
	VXOR	OUTPERM, TMP, OUTPERM              // vxor v9, v3, v9
	LVX	(IVP)(R0), OUTHEAD                 // lvx v7,r0,r7
	VPERM	IVEC, IVEC, OUTPERM, IVEC          // vperm v4,v4,v4,v8
	VSEL	OUTHEAD, IVEC, OUTMASK, INOUT      // vsel v2,v7,v4,v9
	LVX	(IVP)(IDX), INPTAIL                // lvx v5,r10,r7
	STVX	INOUT, (IVP)(R0)                   // stvx v2,r0,r7
	VSEL	IVEC, INPTAIL, OUTMASK, INOUT      // vsel v2,v4,v5,v9
	STVX	INOUT, (IVP)(IDX)                  // stvx v2,r10,r7
	RET                                        // bclr 20,lt,0


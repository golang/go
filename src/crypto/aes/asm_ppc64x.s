// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (ppc64 || ppc64le) && !purego

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
#define TMP2    V22  // Temporary for P8_STXVB16X/P8_STXVB16X

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

#ifdef GOARCH_ppc64le
#  ifdef GOPPC64_power9
#define P8_LXVB16X(RA,RB,VT)  LXVB16X	(RA+RB), VT
#define P8_STXVB16X(VS,RA,RB) STXVB16X	VS, (RA+RB)
#define XXBRD_ON_LE(VA,VT)    XXBRD	VA, VT
#  else
// On POWER8/ppc64le, emulate the POWER9 instructions by loading unaligned
// doublewords and byte-swapping each doubleword to emulate BE load/stores.
#define NEEDS_ESPERM
#define P8_LXVB16X(RA,RB,VT) \
	LXVD2X	(RA+RB), VT \
	VPERM	VT, VT, ESPERM, VT

#define P8_STXVB16X(VS,RA,RB) \
	VPERM	VS, VS, ESPERM, TMP2 \
	STXVD2X	TMP2, (RA+RB)

#define XXBRD_ON_LE(VA,VT) \
	VPERM	VA, VA, ESPERM, VT

#  endif // defined(GOPPC64_power9)
#else
#define P8_LXVB16X(RA,RB,VT)  LXVD2X	(RA+RB), VT
#define P8_STXVB16X(VS,RA,RB) STXVD2X	VS, (RA+RB)
#define XXBRD_ON_LE(VA, VT)
#endif // defined(GOARCH_ppc64le)

// func setEncryptKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)
TEXT ·expandKeyAsm(SB), NOSPLIT|NOFRAME, $0
	// Load the arguments inside the registers
	MOVD	nr+0(FP), ROUNDS
	MOVD	key+8(FP), INP
	MOVD	enc+16(FP), OUTENC
	MOVD	dec+24(FP), OUTDEC

#ifdef NEEDS_ESPERM
	MOVD	$·rcon(SB), PTR // PTR points to rcon addr
	LVX	(PTR), ESPERM
	ADD	$0x10, PTR
#else
	MOVD	$·rcon+0x10(SB), PTR // PTR points to rcon addr (skipping permute vector)
#endif

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
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)
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
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)
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
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)
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
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)

	RET

l192:
	LXSDX	(INP+R0), IN1                    // Load next 8 bytes into upper half of VSR.
	XXBRD_ON_LE(IN1, IN1)                    // and convert to BE ordering on LE hosts.
	MOVD	$4, CNT                          // li 7,4
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)
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
	STXVD2X	STAGE, (R0+OUTENC)
	STXVD2X	STAGE, (R0+OUTDEC)
	VCIPHERLAST	KEY, RCON, KEY           // vcipherlast 3,3,4
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC

	VSLDOI	$8, IN0, IN1, STAGE              // vsldoi 7,1,2,8
	VXOR	IN0, TMP, IN0                    // vxor 1,1,6
	VSLDOI	$12, ZERO, TMP, TMP              // vsldoi 6,0,6,12
	STXVD2X	STAGE, (R0+OUTENC)
	STXVD2X	STAGE, (R0+OUTDEC)
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
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC
	BC	0x10, 0, loop192                 // bdnz .Loop192

	RET

l256:
	P8_LXVB16X(INP, R0, IN1)
	MOVD	$7, CNT                          // li 7,7
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)
	ADD	$16, OUTENC, OUTENC
	ADD	$-16, OUTDEC, OUTDEC
	MOVD	CNT, CTR                         // mtctr 7

loop256:
	VPERM	IN1, IN1, MASK, KEY              // vperm 3,2,2,5
	VSLDOI	$12, ZERO, IN0, TMP              // vsldoi 6,0,1,12
	STXVD2X	IN1, (R0+OUTENC)
	STXVD2X	IN1, (R0+OUTDEC)
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
	STXVD2X	IN0, (R0+OUTENC)
	STXVD2X	IN0, (R0+OUTDEC)
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
	MOVD	nr+0(FP), R6   // Round count/Key size
	MOVD	xk+8(FP), R5   // Key pointer
	MOVD	dst+16(FP), R3 // Dest pointer
	MOVD	src+24(FP), R4 // Src pointer
#ifdef NEEDS_ESPERM
	MOVD	$·rcon(SB), R7
	LVX	(R7), ESPERM   // Permute value for P8_ macros.
#endif

	// Set CR{1,2,3}EQ to hold the key size information.
	CMPU	R6, $10, CR1
	CMPU	R6, $12, CR2
	CMPU	R6, $14, CR3

	MOVD	$16, R6
	MOVD	$32, R7
	MOVD	$48, R8
	MOVD	$64, R9
	MOVD	$80, R10
	MOVD	$96, R11
	MOVD	$112, R12

	// Load text in BE order
	P8_LXVB16X(R4, R0, V0)

	// V1, V2 will hold keys, V0 is a temp.
	// At completion, V2 will hold the ciphertext.
	// Load xk[0:3] and xor with text
	LXVD2X	(R0+R5), V1
	VXOR	V0, V1, V0

	// Load xk[4:11] and cipher
	LXVD2X	(R6+R5), V1
	LXVD2X	(R7+R5), V2
	VCIPHER	V0, V1, V0
	VCIPHER	V0, V2, V0

	// Load xk[12:19] and cipher
	LXVD2X	(R8+R5), V1
	LXVD2X	(R9+R5), V2
	VCIPHER	V0, V1, V0
	VCIPHER	V0, V2, V0

	// Load xk[20:27] and cipher
	LXVD2X	(R10+R5), V1
	LXVD2X	(R11+R5), V2
	VCIPHER	V0, V1, V0
	VCIPHER	V0, V2, V0

	// Increment xk pointer to reuse constant offsets in R6-R12.
	ADD	$112, R5

	// Load xk[28:35] and cipher
	LXVD2X	(R0+R5), V1
	LXVD2X	(R6+R5), V2
	VCIPHER	V0, V1, V0
	VCIPHER	V0, V2, V0

	// Load xk[36:43] and cipher
	LXVD2X	(R7+R5), V1
	LXVD2X	(R8+R5), V2
	BEQ	CR1, Ldec_tail // Key size 10?
	VCIPHER	V0, V1, V0
	VCIPHER	V0, V2, V0

	// Load xk[44:51] and cipher
	LXVD2X	(R9+R5), V1
	LXVD2X	(R10+R5), V2
	BEQ	CR2, Ldec_tail // Key size 12?
	VCIPHER	V0, V1, V0
	VCIPHER	V0, V2, V0

	// Load xk[52:59] and cipher
	LXVD2X	(R11+R5), V1
	LXVD2X	(R12+R5), V2
	BNE	CR3, Linvalid_key_len // Not key size 14?
	// Fallthrough to final cipher

Ldec_tail:
	// Cipher last two keys such that key information is
	// cleared from V1 and V2.
	VCIPHER		V0, V1, V1
	VCIPHERLAST	V1, V2, V2

	// Store the result in BE order.
	P8_STXVB16X(V2, R3, R0)
	RET

Linvalid_key_len:
	// Segfault, this should never happen. Only 3 keys sizes are created/used.
	MOVD	R0, 0(R0)
	RET

// func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)
TEXT ·decryptBlockAsm(SB), NOSPLIT|NOFRAME, $0
	MOVD	nr+0(FP), R6   // Round count/Key size
	MOVD	xk+8(FP), R5   // Key pointer
	MOVD	dst+16(FP), R3 // Dest pointer
	MOVD	src+24(FP), R4 // Src pointer
#ifdef NEEDS_ESPERM
	MOVD	$·rcon(SB), R7
	LVX	(R7), ESPERM   // Permute value for P8_ macros.
#endif

	// Set CR{1,2,3}EQ to hold the key size information.
	CMPU	R6, $10, CR1
	CMPU	R6, $12, CR2
	CMPU	R6, $14, CR3

	MOVD	$16, R6
	MOVD	$32, R7
	MOVD	$48, R8
	MOVD	$64, R9
	MOVD	$80, R10
	MOVD	$96, R11
	MOVD	$112, R12

	// Load text in BE order
	P8_LXVB16X(R4, R0, V0)

	// V1, V2 will hold keys, V0 is a temp.
	// At completion, V2 will hold the text.
	// Load xk[0:3] and xor with ciphertext
	LXVD2X	(R0+R5), V1
	VXOR	V0, V1, V0

	// Load xk[4:11] and cipher
	LXVD2X	(R6+R5), V1
	LXVD2X	(R7+R5), V2
	VNCIPHER	V0, V1, V0
	VNCIPHER	V0, V2, V0

	// Load xk[12:19] and cipher
	LXVD2X	(R8+R5), V1
	LXVD2X	(R9+R5), V2
	VNCIPHER	V0, V1, V0
	VNCIPHER	V0, V2, V0

	// Load xk[20:27] and cipher
	LXVD2X	(R10+R5), V1
	LXVD2X	(R11+R5), V2
	VNCIPHER	V0, V1, V0
	VNCIPHER	V0, V2, V0

	// Increment xk pointer to reuse constant offsets in R6-R12.
	ADD	$112, R5

	// Load xk[28:35] and cipher
	LXVD2X	(R0+R5), V1
	LXVD2X	(R6+R5), V2
	VNCIPHER	V0, V1, V0
	VNCIPHER	V0, V2, V0

	// Load xk[36:43] and cipher
	LXVD2X	(R7+R5), V1
	LXVD2X	(R8+R5), V2
	BEQ	CR1, Ldec_tail // Key size 10?
	VNCIPHER	V0, V1, V0
	VNCIPHER	V0, V2, V0

	// Load xk[44:51] and cipher
	LXVD2X	(R9+R5), V1
	LXVD2X	(R10+R5), V2
	BEQ	CR2, Ldec_tail // Key size 12?
	VNCIPHER	V0, V1, V0
	VNCIPHER	V0, V2, V0

	// Load xk[52:59] and cipher
	LXVD2X	(R11+R5), V1
	LXVD2X	(R12+R5), V2
	BNE	CR3, Linvalid_key_len // Not key size 14?
	// Fallthrough to final cipher

Ldec_tail:
	// Cipher last two keys such that key information is
	// cleared from V1 and V2.
	VNCIPHER	V0, V1, V1
	VNCIPHERLAST	V1, V2, V2

	// Store the result in BE order.
	P8_STXVB16X(V2, R3, R0)
	RET

Linvalid_key_len:
	// Segfault, this should never happen. Only 3 keys sizes are created/used.
	MOVD	R0, 0(R0)
	RET

// Remove defines from above so they can be defined here
#undef INP
#undef OUTENC
#undef ROUNDS
#undef KEY
#undef TMP

#define INP R3
#define OUTP R4
#define LEN R5
#define KEYP R6
#define ROUNDS R7
#define IVP R8
#define ENC R9

#define INOUT V2
#define TMP V3
#define IVEC V4

// Load the crypt key into VSRs.
//
// The expanded key is stored and loaded using
// STXVD2X/LXVD2X. The in-memory byte ordering
// depends on the endianness of the machine. The
// expanded keys are generated by expandKeyAsm above.
//
// Rkeyp holds the key pointer. It is clobbered. Once
// the expanded keys are loaded, it is not needed.
//
// R12,R14-R21 are scratch registers.
// For keyp of 10, V6, V11-V20 hold the expanded key.
// For keyp of 12, V6, V9-V20 hold the expanded key.
// For keyp of 14, V6, V7-V20 hold the expanded key.
#define LOAD_KEY(Rkeyp) \
	MOVD	$16, R12 \
	MOVD	$32, R14 \
	MOVD	$48, R15 \
	MOVD	$64, R16 \
	MOVD	$80, R17 \
	MOVD	$96, R18 \
	MOVD	$112, R19 \
	MOVD	$128, R20 \
	MOVD	$144, R21 \
	LXVD2X	(R0+Rkeyp), V6 \
	ADD	$16, Rkeyp \
	BEQ	CR1, L_start10 \
	BEQ	CR2, L_start12 \
	LXVD2X	(R0+Rkeyp), V7 \
	LXVD2X	(R12+Rkeyp), V8 \
	ADD	$32, Rkeyp \
	L_start12: \
	LXVD2X	(R0+Rkeyp), V9 \
	LXVD2X	(R12+Rkeyp), V10 \
	ADD	$32, Rkeyp \
	L_start10: \
	LXVD2X	(R0+Rkeyp), V11 \
	LXVD2X	(R12+Rkeyp), V12 \
	LXVD2X	(R14+Rkeyp), V13 \
	LXVD2X	(R15+Rkeyp), V14 \
	LXVD2X	(R16+Rkeyp), V15 \
	LXVD2X	(R17+Rkeyp), V16 \
	LXVD2X	(R18+Rkeyp), V17 \
	LXVD2X	(R19+Rkeyp), V18 \
	LXVD2X	(R20+Rkeyp), V19 \
	LXVD2X	(R21+Rkeyp), V20

// Perform aes cipher operation for keysize 10/12/14 using the keys
// loaded by LOAD_KEY, and key size information held in CR1EQ/CR2EQ.
//
// Vxor is ideally V6 (Key[0-3]), but for slightly improved encrypting
// performance V6 and IVEC can be swapped (xor is both associative and
// commutative) during encryption:
//
//	VXOR INOUT, IVEC, INOUT
//	VXOR INOUT, V6, INOUT
//
//	into
//
//	VXOR INOUT, V6, INOUT
//	VXOR INOUT, IVEC, INOUT
//
#define CIPHER_BLOCK(Vin, Vxor, Vout, vcipher, vciphel, label10, label12) \
	VXOR	Vin, Vxor, Vout \
	BEQ	CR1, label10 \
	BEQ	CR2, label12 \
	vcipher	Vout, V7, Vout \
	vcipher	Vout, V8, Vout \
	label12: \
	vcipher	Vout, V9, Vout \
	vcipher	Vout, V10, Vout \
	label10: \
	vcipher	Vout, V11, Vout \
	vcipher	Vout, V12, Vout \
	vcipher	Vout, V13, Vout \
	vcipher	Vout, V14, Vout \
	vcipher	Vout, V15, Vout \
	vcipher	Vout, V16, Vout \
	vcipher	Vout, V17, Vout \
	vcipher	Vout, V18, Vout \
	vcipher	Vout, V19, Vout \
	vciphel	Vout, V20, Vout \

#define CLEAR_KEYS() \
	VXOR	V6, V6, V6 \
	VXOR	V7, V7, V7 \
	VXOR	V8, V8, V8 \
	VXOR	V9, V9, V9 \
	VXOR	V10, V10, V10 \
	VXOR	V11, V11, V11 \
	VXOR	V12, V12, V12 \
	VXOR	V13, V13, V13 \
	VXOR	V14, V14, V14 \
	VXOR	V15, V15, V15 \
	VXOR	V16, V16, V16 \
	VXOR	V17, V17, V17 \
	VXOR	V18, V18, V18 \
	VXOR	V19, V19, V19 \
	VXOR	V20, V20, V20

//func cryptBlocksChain(src, dst *byte, length int, key *uint32, iv *byte, enc int, nr int)
TEXT ·cryptBlocksChain(SB), NOSPLIT|NOFRAME, $0
	MOVD	src+0(FP), INP
	MOVD	dst+8(FP), OUTP
	MOVD	length+16(FP), LEN
	MOVD	key+24(FP), KEYP
	MOVD	iv+32(FP), IVP
	MOVD	enc+40(FP), ENC
	MOVD	nr+48(FP), ROUNDS

#ifdef NEEDS_ESPERM
	MOVD	$·rcon(SB), R11
	LVX	(R11), ESPERM   // Permute value for P8_ macros.
#endif

	// Assume len > 0 && len % blockSize == 0.
	CMPW	ENC, $0
	P8_LXVB16X(IVP, R0, IVEC)
	CMPU	ROUNDS, $10, CR1
	CMPU	ROUNDS, $12, CR2 // Only sizes 10/12/14 are supported.

	// Setup key in VSRs, and set loop count in CTR.
	LOAD_KEY(KEYP)
	SRD	$4, LEN
	MOVD	LEN, CTR

	BEQ	Lcbc_dec

	PCALIGN $16
Lcbc_enc:
	P8_LXVB16X(INP, R0, INOUT)
	ADD	$16, INP
	VXOR	INOUT, V6, INOUT
	CIPHER_BLOCK(INOUT, IVEC, INOUT, VCIPHER, VCIPHERLAST, Lcbc_enc10, Lcbc_enc12)
	VOR	INOUT, INOUT, IVEC // ciphertext (INOUT) is IVEC for next block.
	P8_STXVB16X(INOUT, OUTP, R0)
	ADD	$16, OUTP
	BDNZ	Lcbc_enc

	P8_STXVB16X(INOUT, IVP, R0)
	CLEAR_KEYS()
	RET

	PCALIGN $16
Lcbc_dec:
	P8_LXVB16X(INP, R0, TMP)
	ADD	$16, INP
	CIPHER_BLOCK(TMP, V6, INOUT, VNCIPHER, VNCIPHERLAST, Lcbc_dec10, Lcbc_dec12)
	VXOR	INOUT, IVEC, INOUT
	VOR	TMP, TMP, IVEC // TMP is IVEC for next block.
	P8_STXVB16X(INOUT, OUTP, R0)
	ADD	$16, OUTP
	BDNZ	Lcbc_dec

	P8_STXVB16X(IVEC, IVP, R0)
	CLEAR_KEYS()
	RET

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gc,!purego

#include "textflag.h"

// This was ported from the amd64 implementation.

#define POLY1305_ADD(msg, h0, h1, h2, t0, t1, t2) \
	MOVD (msg), t0;  \
	MOVD 8(msg), t1; \
	MOVD $1, t2;     \
	ADDC t0, h0, h0; \
	ADDE t1, h1, h1; \
	ADDE t2, h2;     \
	ADD  $16, msg

#define POLY1305_MUL(h0, h1, h2, r0, r1, t0, t1, t2, t3, t4, t5) \
	MULLD  r0, h0, t0;  \
	MULLD  r0, h1, t4;  \
	MULHDU r0, h0, t1;  \
	MULHDU r0, h1, t5;  \
	ADDC   t4, t1, t1;  \
	MULLD  r0, h2, t2;  \
	ADDZE  t5;          \
	MULHDU r1, h0, t4;  \
	MULLD  r1, h0, h0;  \
	ADD    t5, t2, t2;  \
	ADDC   h0, t1, t1;  \
	MULLD  h2, r1, t3;  \
	ADDZE  t4, h0;      \
	MULHDU r1, h1, t5;  \
	MULLD  r1, h1, t4;  \
	ADDC   t4, t2, t2;  \
	ADDE   t5, t3, t3;  \
	ADDC   h0, t2, t2;  \
	MOVD   $-4, t4;     \
	MOVD   t0, h0;      \
	MOVD   t1, h1;      \
	ADDZE  t3;          \
	ANDCC  $3, t2, h2;  \
	AND    t2, t4, t0;  \
	ADDC   t0, h0, h0;  \
	ADDE   t3, h1, h1;  \
	SLD    $62, t3, t4; \
	SRD    $2, t2;      \
	ADDZE  h2;          \
	OR     t4, t2, t2;  \
	SRD    $2, t3;      \
	ADDC   t2, h0, h0;  \
	ADDE   t3, h1, h1;  \
	ADDZE  h2

DATA ·poly1305Mask<>+0x00(SB)/8, $0x0FFFFFFC0FFFFFFF
DATA ·poly1305Mask<>+0x08(SB)/8, $0x0FFFFFFC0FFFFFFC
GLOBL ·poly1305Mask<>(SB), RODATA, $16

// func update(state *[7]uint64, msg []byte)
TEXT ·update(SB), $0-32
	MOVD state+0(FP), R3
	MOVD msg_base+8(FP), R4
	MOVD msg_len+16(FP), R5

	MOVD 0(R3), R8   // h0
	MOVD 8(R3), R9   // h1
	MOVD 16(R3), R10 // h2
	MOVD 24(R3), R11 // r0
	MOVD 32(R3), R12 // r1

	CMP R5, $16
	BLT bytes_between_0_and_15

loop:
	POLY1305_ADD(R4, R8, R9, R10, R20, R21, R22)

multiply:
	POLY1305_MUL(R8, R9, R10, R11, R12, R16, R17, R18, R14, R20, R21)
	ADD $-16, R5
	CMP R5, $16
	BGE loop

bytes_between_0_and_15:
	CMP  R5, $0
	BEQ  done
	MOVD $0, R16 // h0
	MOVD $0, R17 // h1

flush_buffer:
	CMP R5, $8
	BLE just1

	MOVD $8, R21
	SUB  R21, R5, R21

	// Greater than 8 -- load the rightmost remaining bytes in msg
	// and put into R17 (h1)
	MOVD (R4)(R21), R17
	MOVD $16, R22

	// Find the offset to those bytes
	SUB R5, R22, R22
	SLD $3, R22

	// Shift to get only the bytes in msg
	SRD R22, R17, R17

	// Put 1 at high end
	MOVD $1, R23
	SLD  $3, R21
	SLD  R21, R23, R23
	OR   R23, R17, R17

	// Remainder is 8
	MOVD $8, R5

just1:
	CMP R5, $8
	BLT less8

	// Exactly 8
	MOVD (R4), R16

	CMP R17, $0

	// Check if we've already set R17; if not
	// set 1 to indicate end of msg.
	BNE  carry
	MOVD $1, R17
	BR   carry

less8:
	MOVD  $0, R16   // h0
	MOVD  $0, R22   // shift count
	CMP   R5, $4
	BLT   less4
	MOVWZ (R4), R16
	ADD   $4, R4
	ADD   $-4, R5
	MOVD  $32, R22

less4:
	CMP   R5, $2
	BLT   less2
	MOVHZ (R4), R21
	SLD   R22, R21, R21
	OR    R16, R21, R16
	ADD   $16, R22
	ADD   $-2, R5
	ADD   $2, R4

less2:
	CMP   R5, $0
	BEQ   insert1
	MOVBZ (R4), R21
	SLD   R22, R21, R21
	OR    R16, R21, R16
	ADD   $8, R22

insert1:
	// Insert 1 at end of msg
	MOVD $1, R21
	SLD  R22, R21, R21
	OR   R16, R21, R16

carry:
	// Add new values to h0, h1, h2
	ADDC  R16, R8
	ADDE  R17, R9
	ADDZE R10, R10
	MOVD  $16, R5
	ADD   R5, R4
	BR    multiply

done:
	// Save h0, h1, h2 in state
	MOVD R8, 0(R3)
	MOVD R9, 8(R3)
	MOVD R10, 16(R3)
	RET

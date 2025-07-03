// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc && !purego

// func update(state *macState, msg []byte)
TEXT ·update(SB), $0-32
	MOVV	state+0(FP), R4
	MOVV	msg_base+8(FP), R5
	MOVV	msg_len+16(FP), R6

	MOVV	$0x10, R7

	MOVV	(R4), R8	// h0
	MOVV	8(R4), R9	// h1
	MOVV	16(R4), R10	// h2
	MOVV	24(R4), R11	// r0
	MOVV	32(R4), R12	// r1

	BLT	R6, R7, bytes_between_0_and_15

loop:
	MOVV	(R5), R14	// msg[0:8]
	MOVV	8(R5), R16	// msg[8:16]
	ADDV	R14, R8, R8	// h0 (x1 + y1 = z1', if z1' < x1 then z1' overflow)
	ADDV	R16, R9, R27
	SGTU	R14, R8, R24	// h0.carry
	SGTU	R9, R27, R28
	ADDV	R27, R24, R9	// h1
	SGTU	R27, R9, R24
	OR	R24, R28, R24	// h1.carry
	ADDV	$0x01, R24, R24
	ADDV	R10, R24, R10	// h2

	ADDV	$16, R5, R5	// msg = msg[16:]

multiply:
	MULV	R8, R11, R14	// h0r0.lo
	MULHVU	R8, R11, R15	// h0r0.hi
	MULV	R9, R11, R13	// h1r0.lo
	MULHVU	R9, R11, R16	// h1r0.hi
	ADDV	R13, R15, R15
	SGTU	R13, R15, R24
	ADDV	R24, R16, R16
	MULV	R10, R11, R25
	ADDV	R16, R25, R25
	MULV	R8, R12, R13	// h0r1.lo
	MULHVU	R8, R12, R16	// h0r1.hi
	ADDV	R13, R15, R15
	SGTU	R13, R15, R24
	ADDV	R24, R16, R16
	MOVV	R16, R8
	MULV	R10, R12, R26	// h2r1
	MULV	R9, R12, R13	// h1r1.lo
	MULHVU	R9, R12, R16	// h1r1.hi
	ADDV	R13, R25, R25
	ADDV	R16, R26, R27
	SGTU	R13, R25, R24
	ADDV	R27, R24, R26
	ADDV	R8, R25, R25
	SGTU	R8, R25, R24
	ADDV	R24, R26, R26
	AND	$3, R25, R10
	AND	$-4, R25, R17
	ADDV	R17, R14, R8
	ADDV	R26, R15, R27
	SGTU	R17, R8, R24
	SGTU	R26, R27, R28
	ADDV	R27, R24, R9
	SGTU	R27, R9, R24
	OR	R24, R28, R24
	ADDV	R24, R10, R10
	SLLV	$62, R26, R27
	SRLV	$2, R25, R28
	SRLV	$2, R26, R26
	OR	R27, R28, R25
	ADDV	R25, R8, R8
	ADDV	R26, R9, R27
	SGTU	R25, R8, R24
	SGTU	R26, R27, R28
	ADDV	R27, R24, R9
	SGTU	R27, R9, R24
	OR	R24, R28, R24
	ADDV	R24, R10, R10

	SUBV	$16, R6, R6
	BGE	R6, R7, loop

bytes_between_0_and_15:
	BEQ	R6, R0, done
	MOVV	$1, R14
	XOR	R15, R15
	ADDV	R6, R5, R5

flush_buffer:
	MOVBU	-1(R5), R25
	SRLV	$56, R14, R24
	SLLV	$8, R15, R28
	SLLV	$8, R14, R14
	OR	R24, R28, R15
	XOR	R25, R14, R14
	SUBV	$1, R6, R6
	SUBV	$1, R5, R5
	BNE	R6, R0, flush_buffer

	ADDV	R14, R8, R8
	SGTU	R14, R8, R24
	ADDV	R15, R9, R27
	SGTU	R15, R27, R28
	ADDV	R27, R24, R9
	SGTU	R27, R9, R24
	OR	R24, R28, R24
	ADDV	R10, R24, R10

	MOVV	$16, R6
	JMP	multiply

done:
	MOVV	R8, (R4)
	MOVV	R9, 8(R4)
	MOVV	R10, 16(R4)
	RET

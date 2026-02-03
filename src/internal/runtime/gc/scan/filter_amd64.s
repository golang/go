// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·FilterNilAVX512(SB), NOSPLIT, $0-20
	// Load arguments
	MOVQ bufp+0(FP), R8	// R8 = bufp (start of the uint64 array)
	MOVL n+8(FP), R9	// R9 = n (total length)
	XORL R10, R10		// R10 = 0 (scanned = 0)
	XORL R11, R11		// R11 = 0 (cnt = 0)

	MOVL R9, R12	// R12 = n
	SUBL R10, R12	// R12 = n - scanned
	CMPL R12, $8	// Compare (n - scanned) with 8
	JLT scalar_loop	// If (n - scanned) < 8, jump to the scalar cleanup
	VPXOR X15, X15, X15	// Zero the high bits of Z15

vector_loop:
	LEAQ (R8)(R10*8), R13	// R13 = buf[scanned:] address
	VMOVDQU64 (R13), Z1		// Z1 = v (Load 8 uint64s)
	VPCMPUQ $4, Z1, Z15, K1	// Z15 is always 0, compare Z1 with 0, results in K1.

	LEAQ (R8)(R11*8), R14	// R14 = buf[cnt:] address
	VPCOMPRESSQ Z1, K1, Z1	// compress v
	VMOVDQU64 Z1, (R14)		// store v to buf[cnt:]

	KMOVW K1, R15
	POPCNTL R15, R15	// R15 = popcount(K1)

	ADDL R15, R11	// cnt += popcount(K1)
	ADDL $8, R10	// scanned += 8

	MOVL R9, R12	// R12 = n
	SUBL R10, R12	// R12 = n - scanned
	CMPL R12, $8	// Compare (n - scanned) with 8
	JGE vector_loop	// If (n - scanned) >= 8, continue loop

scalar_loop:
	CMPL R10, R9	// Compare scanned with n
	JGE end			// If scanned >= n, loop is done

scalar_next_i:
	LEAQ (R8)(R10*8), R13	// R13 = &buf[scanned]
	MOVQ (R13), R14			// R14 = buf[scanned]

	CMPQ R14, $0
	JE scalar_increment_i	// If buf[i] == 0, skip to increment i

	LEAQ (R8)(R11*8), R15	// R15 = &buf[cnt]
	MOVQ R14, (R15)			// buf[cnt] = buf[scanned]

	ADDL $1, R11	// cnt++

scalar_increment_i:
	ADDL $1, R10	// scanned++

	CMPL R10, R9
	JL scalar_next_i	// if scanned < n, continue

end:
	MOVL R11, ret+16(FP)
	VZEROUPPER
	RET

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

/*
 * void crosscall2(void (*fn)(void*, int32), void*, int32)
 * Save registers and call fn with two arguments.
 * crosscall2 obeys the C ABI; fn obeys the Go ABI.
 */
TEXT crosscall2(SB),NOSPLIT,$-8
	// TODO(austin): ABI v1 (fn is probably a function descriptor)

	// Start with standard C stack frame layout and linkage
	MOVD	LR, R0
	MOVD	R0, 16(R1)	// Save LR in caller's frame
	MOVD	R2, 24(R1)	// Save TOC in caller's frame

	BL	saveregs2<>(SB)

	MOVDU	R1, (-288-3*8)(R1)

	// Initialize Go ABI environment
	BL	runtime·reginit(SB)
	BL	runtime·load_g(SB)

	MOVD	R3, CTR
	MOVD	R4, 8(R1)
	MOVD	R5, 16(R1)
	BL	(CTR)

	ADD	$(288+3*8), R1

	BL	restoreregs2<>(SB)

	MOVD	24(R1), R2
	MOVD	16(R1), R0
	MOVD	R0, LR
	RET

TEXT saveregs2<>(SB),NOSPLIT,$-8
	// O=-288; for R in R{14..31}; do echo "\tMOVD\t$R, $O(R1)"|sed s/R30/g/; ((O+=8)); done; for F in F{14..31}; do echo "\tFMOVD\t$F, $O(R1)"; ((O+=8)); done
	MOVD	R14, -288(R1)
	MOVD	R15, -280(R1)
	MOVD	R16, -272(R1)
	MOVD	R17, -264(R1)
	MOVD	R18, -256(R1)
	MOVD	R19, -248(R1)
	MOVD	R20, -240(R1)
	MOVD	R21, -232(R1)
	MOVD	R22, -224(R1)
	MOVD	R23, -216(R1)
	MOVD	R24, -208(R1)
	MOVD	R25, -200(R1)
	MOVD	R26, -192(R1)
	MOVD	R27, -184(R1)
	MOVD	R28, -176(R1)
	MOVD	R29, -168(R1)
	MOVD	g, -160(R1)
	MOVD	R31, -152(R1)
	FMOVD	F14, -144(R1)
	FMOVD	F15, -136(R1)
	FMOVD	F16, -128(R1)
	FMOVD	F17, -120(R1)
	FMOVD	F18, -112(R1)
	FMOVD	F19, -104(R1)
	FMOVD	F20, -96(R1)
	FMOVD	F21, -88(R1)
	FMOVD	F22, -80(R1)
	FMOVD	F23, -72(R1)
	FMOVD	F24, -64(R1)
	FMOVD	F25, -56(R1)
	FMOVD	F26, -48(R1)
	FMOVD	F27, -40(R1)
	FMOVD	F28, -32(R1)
	FMOVD	F29, -24(R1)
	FMOVD	F30, -16(R1)
	FMOVD	F31, -8(R1)

	RET

TEXT restoreregs2<>(SB),NOSPLIT,$-8
	// O=-288; for R in R{14..31}; do echo "\tMOVD\t$O(R1), $R"|sed s/R30/g/; ((O+=8)); done; for F in F{14..31}; do echo "\tFMOVD\t$O(R1), $F"; ((O+=8)); done
	MOVD	-288(R1), R14
	MOVD	-280(R1), R15
	MOVD	-272(R1), R16
	MOVD	-264(R1), R17
	MOVD	-256(R1), R18
	MOVD	-248(R1), R19
	MOVD	-240(R1), R20
	MOVD	-232(R1), R21
	MOVD	-224(R1), R22
	MOVD	-216(R1), R23
	MOVD	-208(R1), R24
	MOVD	-200(R1), R25
	MOVD	-192(R1), R26
	MOVD	-184(R1), R27
	MOVD	-176(R1), R28
	MOVD	-168(R1), R29
	MOVD	-160(R1), g
	MOVD	-152(R1), R31
	FMOVD	-144(R1), F14
	FMOVD	-136(R1), F15
	FMOVD	-128(R1), F16
	FMOVD	-120(R1), F17
	FMOVD	-112(R1), F18
	FMOVD	-104(R1), F19
	FMOVD	-96(R1), F20
	FMOVD	-88(R1), F21
	FMOVD	-80(R1), F22
	FMOVD	-72(R1), F23
	FMOVD	-64(R1), F24
	FMOVD	-56(R1), F25
	FMOVD	-48(R1), F26
	FMOVD	-40(R1), F27
	FMOVD	-32(R1), F28
	FMOVD	-24(R1), F29
	FMOVD	-16(R1), F30
	FMOVD	-8(R1), F31

	RET

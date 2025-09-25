// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Note: this assembly file is used for testing only.
// We need to access registers directly to properly test
// that secrets are erased and go test doesn't like to conditionally
// include assembly files.
// These functions defined in the package proper and we
// rely on the linker to prune these away in regular builds

#include "go_asm.h"
#include "funcdata.h"

TEXT ·loadRegisters(SB),0,$0-8
	MOVD	p+0(FP), R0

	MOVD	(R0), R10
	MOVD	(R0), R11
	MOVD	(R0), R12
	MOVD	(R0), R13

	FMOVD	(R0), F15
	FMOVD	(R0), F16
	FMOVD	(R0), F17
	FMOVD	(R0), F18

	VLD1	(R0), [V20.B16]
	VLD1	(R0), [V21.H8]
	VLD1	(R0), [V22.S4]
	VLD1	(R0), [V23.D2]

	RET

TEXT ·spillRegisters(SB),0,$0-16
	MOVD	p+0(FP), R0
	MOVD	R0, R1

	MOVD	R10, (R0)
	MOVD	R11, 8(R0)
	MOVD	R12, 16(R0)
	MOVD	R13, 24(R0)
	ADD	$32, R0

	FMOVD	F15, (R0)
	FMOVD	F16, 16(R0)
	FMOVD	F17, 32(R0)
	FMOVD	F18, 64(R0)
	ADD	$64, R0

	VST1.P	[V20.B16], (R0)
	VST1.P	[V21.H8], (R0)
	VST1.P	[V22.S4], (R0)
	VST1.P	[V23.D2], (R0)

	SUB	R1, R0, R0
	MOVD	R0, ret+8(FP)
	RET

TEXT ·useSecret(SB),0,$0-24
	NO_LOCAL_POINTERS

	// Load secret into R0
	MOVD	secret_base+0(FP), R0
	MOVD	(R0), R0
	// Scatter secret across registers.
	// Increment low byte so we can tell which register
	// a leaking secret came from.

	// TODO(dmo): more substantial dirtying here
	ADD	$1, R0
	MOVD	R0, R1
	ADD	$1, R0
	MOVD	R0, R2
	ADD	$1, R0
	MOVD	R0, R3
	ADD	$1, R0
	MOVD	R0, R4
	ADD	$1, R0
	MOVD	R0, R5
	ADD	$1, R0
	MOVD	R0, R6
	ADD	$1, R0
	MOVD	R0, R7
	ADD	$1, R0
	MOVD	R0, R8
	ADD	$1, R0
	MOVD	R0, R9
	ADD	$1, R0
	MOVD	R0, R10
	ADD	$1, R0
	MOVD	R0, R11
	ADD	$1, R0
	MOVD	R0, R12
	ADD	$1, R0
	MOVD	R0, R13
	ADD	$1, R0
	MOVD	R0, R14
	ADD	$1, R0
	MOVD	R0, R15

	// Dirty the floating point registers
	ADD     $1, R0
	FMOVD   R0, F0
	ADD     $1, R0
	FMOVD   R0, F1
	ADD     $1, R0
	FMOVD   R0, F2
	ADD     $1, R0
	FMOVD   R0, F3
	ADD     $1, R0
	FMOVD   R0, F4
	ADD     $1, R0
	FMOVD   R0, F5
	ADD     $1, R0
	FMOVD   R0, F6
	ADD     $1, R0
	FMOVD   R0, F7
	ADD     $1, R0
	FMOVD   R0, F8
	ADD     $1, R0
	FMOVD   R0, F9
	ADD     $1, R0
	FMOVD   R0, F10
	ADD     $1, R0
	FMOVD   R0, F11
	ADD     $1, R0
	FMOVD   R0, F12
	ADD     $1, R0
	FMOVD   R0, F13
	ADD     $1, R0
	FMOVD   R0, F14
	ADD     $1, R0
	FMOVD   R0, F15
	ADD     $1, R0
	FMOVD   R0, F16
	ADD     $1, R0
	FMOVD   R0, F17
	ADD     $1, R0
	FMOVD   R0, F18
	ADD     $1, R0
	FMOVD   R0, F19
	ADD     $1, R0
	FMOVD   R0, F20
	ADD     $1, R0
	FMOVD   R0, F21
	ADD     $1, R0
	FMOVD   R0, F22
	ADD     $1, R0
	FMOVD   R0, F23
	ADD     $1, R0
	FMOVD   R0, F24
	ADD     $1, R0
	FMOVD   R0, F25
	ADD     $1, R0
	FMOVD   R0, F26
	ADD     $1, R0
	FMOVD   R0, F27
	ADD     $1, R0
	FMOVD   R0, F28
	ADD     $1, R0
	FMOVD   R0, F29
	ADD     $1, R0
	FMOVD   R0, F30
	ADD     $1, R0
	FMOVD   R0, F31
	RET

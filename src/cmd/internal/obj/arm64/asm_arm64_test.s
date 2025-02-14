// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// testvmovs() (r1, r2 uint64)
TEXT ·testvmovs(SB), NOSPLIT, $0-16
	VMOVS   $0x80402010, V1
	VMOV    V1.D[0], R0
	VMOV    V1.D[1], R1
	MOVD    R0, r1+0(FP)
	MOVD    R1, r2+8(FP)
	RET

// testvmovd() (r1, r2 uint64)
TEXT ·testvmovd(SB), NOSPLIT, $0-16
	VMOVD   $0x7040201008040201, V1
	VMOV    V1.D[0], R0
	VMOV    V1.D[1], R1
	MOVD    R0, r1+0(FP)
	MOVD    R1, r2+8(FP)
	RET

// testvmovq() (r1, r2 uint64)
TEXT ·testvmovq(SB), NOSPLIT, $0-16
	VMOVQ   $0x7040201008040201, $0x3040201008040201, V1
	VMOV    V1.D[0], R0
	VMOV    V1.D[1], R1
	MOVD    R0, r1+0(FP)
	MOVD    R1, r2+8(FP)
	RET

// testmovk() uint64
TEXT ·testmovk(SB), NOSPLIT, $0-8
	MOVD	$0, R0
	MOVK	$(40000<<48), R0
	MOVD	R0, ret+0(FP)
	RET

// testcondbr19() uint64
TEXT ·testcondbr19(SB), NOSPLIT, $0-8
	MOVD	$0, R0
	CBZ R0, ·testcondbr19_target(SB)

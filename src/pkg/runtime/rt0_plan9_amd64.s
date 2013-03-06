// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT _rt0_amd64_plan9(SB),7,$-8
	LEAQ	8(SP), SI // argv
	MOVQ	0(SP), DI // argc
	MOVQ	$_rt0_amd64(SB), AX
	JMP	AX

DATA runtime·isplan9(SB)/4, $1
GLOBL runtime·isplan9(SB), $4

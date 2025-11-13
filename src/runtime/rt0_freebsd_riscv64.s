// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// On FreeBSD argc/argv are passed in R0, not X2
TEXT _rt0_riscv64_freebsd(SB),NOSPLIT|NOFRAME,$0
	ADD	$8, A0, A1	// argv
	MOV	0(A0), A0	// argc
	JMP	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_riscv64_freebsd_lib(SB),NOSPLIT,$0
	JMP	_rt0_riscv64_lib(SB)

TEXT main(SB),NOSPLIT|NOFRAME,$0
	MOV	$runtime·rt0_go(SB), T0
	JALR	ZERO, T0

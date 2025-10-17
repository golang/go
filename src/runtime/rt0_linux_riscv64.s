// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_riscv64_linux(SB),NOSPLIT|NOFRAME,$0
	MOV	0(X2), A0	// argc
	ADD	$8, X2, A1	// argv
	JMP	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_riscv64_linux_lib(SB),NOSPLIT,$0
	JMP	_rt0_riscv64_lib(SB)


TEXT main(SB),NOSPLIT|NOFRAME,$0
	MOV	$runtime·rt0_go(SB), T0
	JALR	ZERO, T0

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
TEXT _rt0_riscv64_linux_lib(SB),NOSPLIT,$224
	// Preserve callee-save registers, along with X1 (LR).
	MOV	X1, (8*3)(X2)
	MOV	X8, (8*4)(X2)
	MOV	X9, (8*5)(X2)
	MOV	X18, (8*6)(X2)
	MOV	X19, (8*7)(X2)
	MOV	X20, (8*8)(X2)
	MOV	X21, (8*9)(X2)
	MOV	X22, (8*10)(X2)
	MOV	X23, (8*11)(X2)
	MOV	X24, (8*12)(X2)
	MOV	X25, (8*13)(X2)
	MOV	X26, (8*14)(X2)
	MOV	g, (8*15)(X2)
	MOVD	F8, (8*16)(X2)
	MOVD	F9, (8*17)(X2)
	MOVD	F18, (8*18)(X2)
	MOVD	F19, (8*19)(X2)
	MOVD	F20, (8*20)(X2)
	MOVD	F21, (8*21)(X2)
	MOVD	F22, (8*22)(X2)
	MOVD	F23, (8*23)(X2)
	MOVD	F24, (8*24)(X2)
	MOVD	F25, (8*25)(X2)
	MOVD	F26, (8*26)(X2)
	MOVD	F27, (8*27)(X2)

	// Initialize g as nil in case of using g later e.g. sigaction in cgo_sigaction.go
	MOV	X0, g

	MOV	A0, _rt0_riscv64_linux_lib_argc<>(SB)
	MOV	A1, _rt0_riscv64_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOV	$runtime·libpreinit(SB), T0
	JALR	RA, T0

	// Create a new thread to do the runtime initialization and return.
	MOV	_cgo_sys_thread_create(SB), T0
	BEQZ	T0, nocgo
	MOV	$_rt0_riscv64_linux_lib_go(SB), A0
	MOV	$0, A1
	JALR	RA, T0
	JMP	restore

nocgo:
	MOV	$0x800000, A0                     // stacksize = 8192KB
	MOV	$_rt0_riscv64_linux_lib_go(SB), A1
	MOV	A0, 8(X2)
	MOV	A1, 16(X2)
	MOV	$runtime·newosproc0(SB), T0
	JALR	RA, T0

restore:
	// Restore callee-save registers, along with X1 (LR).
	MOV	(8*3)(X2), X1
	MOV	(8*4)(X2), X8
	MOV	(8*5)(X2), X9
	MOV	(8*6)(X2), X18
	MOV	(8*7)(X2), X19
	MOV	(8*8)(X2), X20
	MOV	(8*9)(X2), X21
	MOV	(8*10)(X2), X22
	MOV	(8*11)(X2), X23
	MOV	(8*12)(X2), X24
	MOV	(8*13)(X2), X25
	MOV	(8*14)(X2), X26
	MOV	(8*15)(X2), g
	MOVD	(8*16)(X2), F8
	MOVD	(8*17)(X2), F9
	MOVD	(8*18)(X2), F18
	MOVD	(8*19)(X2), F19
	MOVD	(8*20)(X2), F20
	MOVD	(8*21)(X2), F21
	MOVD	(8*22)(X2), F22
	MOVD	(8*23)(X2), F23
	MOVD	(8*24)(X2), F24
	MOVD	(8*25)(X2), F25
	MOVD	(8*26)(X2), F26
	MOVD	(8*27)(X2), F27

	RET

TEXT _rt0_riscv64_linux_lib_go(SB),NOSPLIT,$0
	MOV	_rt0_riscv64_linux_lib_argc<>(SB), A0
	MOV	_rt0_riscv64_linux_lib_argv<>(SB), A1
	MOV	$runtime·rt0_go(SB), T0
	JALR	ZERO, T0

DATA _rt0_riscv64_linux_lib_argc<>(SB)/8, $0
GLOBL _rt0_riscv64_linux_lib_argc<>(SB),NOPTR, $8
DATA _rt0_riscv64_linux_lib_argv<>(SB)/8, $0
GLOBL _rt0_riscv64_linux_lib_argv<>(SB),NOPTR, $8


TEXT main(SB),NOSPLIT|NOFRAME,$0
	MOV	$runtime·rt0_go(SB), T0
	JALR	ZERO, T0

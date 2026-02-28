// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_386_android(SB),NOSPLIT,$0
	JMP	_rt0_386(SB)

TEXT _rt0_386_android_lib(SB),NOSPLIT,$0
	PUSHL	$_rt0_386_android_argv(SB)  // argv
	PUSHL	$1  // argc
	CALL	_rt0_386_lib(SB)
	POPL	AX
	POPL	AX
	RET

DATA _rt0_386_android_argv+0x00(SB)/4,$_rt0_386_android_argv0(SB)
DATA _rt0_386_android_argv+0x04(SB)/4,$0  // argv terminate
DATA _rt0_386_android_argv+0x08(SB)/4,$0  // envp terminate
DATA _rt0_386_android_argv+0x0c(SB)/4,$0  // auxv terminate
GLOBL _rt0_386_android_argv(SB),NOPTR,$0x10

// TODO: wire up necessary VDSO (see os_linux_386.go)

DATA _rt0_386_android_argv0(SB)/8, $"gojni"
GLOBL _rt0_386_android_argv0(SB),RODATA,$8

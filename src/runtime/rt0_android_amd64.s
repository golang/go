// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_amd64_android(SB),NOSPLIT,$-8
	MOVQ	0(SP), DI // argc
	LEAQ	8(SP), SI // argv
	MOVQ	$main(SB), AX
	JMP	AX

TEXT _rt0_amd64_android_lib(SB),NOSPLIT,$0
	MOVQ	$1, DI // argc
	MOVQ	$_rt0_amd64_android_argv(SB), SI  // argv
	MOVQ	$_rt0_amd64_linux_lib(SB), AX
	JMP	AX

DATA _rt0_amd64_android_argv+0x00(SB)/8,$_rt0_amd64_android_argv0(SB)
DATA _rt0_amd64_android_argv+0x08(SB)/8,$0
DATA _rt0_amd64_android_argv+0x10(SB)/8,$0
DATA _rt0_amd64_android_argv+0x18(SB)/8,$15  // AT_PLATFORM
DATA _rt0_amd64_android_argv+0x20(SB)/8,$_rt0_amd64_android_auxv0(SB)
DATA _rt0_amd64_android_argv+0x28(SB)/8,$0
GLOBL _rt0_amd64_android_argv(SB),NOPTR,$0x30

// TODO: AT_HWCAP necessary? If so, what value?

DATA _rt0_amd64_android_argv0(SB)/8, $"gojni"
GLOBL _rt0_amd64_android_argv0(SB),RODATA,$8

DATA _rt0_amd64_android_auxv0(SB)/8, $"x86_64"
GLOBL _rt0_amd64_android_auxv0(SB),RODATA,$8

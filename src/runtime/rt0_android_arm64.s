// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm64_android(SB),NOSPLIT,$-8
	MOVD	$_rt0_arm64_linux(SB), R4
	B	(R4)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_android_lib(SB),NOSPLIT,$-8
	MOVW	$1, R0                            // argc
	MOVD	$_rt0_arm64_android_argv(SB), R1  // **argv
	MOVD	$_rt0_arm64_linux_lib(SB), R4
	B	(R4)

DATA _rt0_arm64_android_argv+0x00(SB)/8,$_rt0_arm64_android_argv0(SB)
DATA _rt0_arm64_android_argv+0x08(SB)/8,$0
DATA _rt0_arm64_android_argv+0x10(SB)/8,$0
GLOBL _rt0_arm64_android_argv(SB),NOPTR,$0x18

DATA _rt0_arm64_android_argv0(SB)/8, $"gojni"
GLOBL _rt0_arm64_android_argv0(SB),RODATA,$8

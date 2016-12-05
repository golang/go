// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_android(SB),NOSPLIT,$-4
	MOVW		(R13), R0      // argc
	MOVW		$4(R13), R1    // argv
	MOVW		$_rt0_arm_linux1(SB), R4
	B		(R4)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm_android_lib(SB),NOSPLIT,$0
	MOVW	$1, R0                          // argc
	MOVW	$_rt0_arm_android_argv(SB), R1  // **argv
	BL _rt0_arm_linux_lib(SB)
	RET

DATA _rt0_arm_android_argv+0x00(SB)/4,$_rt0_arm_android_argv0(SB)
DATA _rt0_arm_android_argv+0x04(SB)/4,$0 // end argv
DATA _rt0_arm_android_argv+0x08(SB)/4,$0 // end envv
DATA _rt0_arm_android_argv+0x0c(SB)/4,$0 // end auxv
GLOBL _rt0_arm_android_argv(SB),NOPTR,$0x10

DATA _rt0_arm_android_argv0(SB)/8, $"gojni"
GLOBL _rt0_arm_android_argv0(SB),RODATA,$8

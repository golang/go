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
DATA _rt0_arm_android_argv+0x04(SB)/4,$0
DATA _rt0_arm_android_argv+0x08(SB)/4,$0
DATA _rt0_arm_android_argv+0x0C(SB)/4,$15      // AT_PLATFORM
DATA _rt0_arm_android_argv+0x10(SB)/4,$_rt0_arm_android_auxv0(SB)
DATA _rt0_arm_android_argv+0x14(SB)/4,$16      // AT_HWCAP
DATA _rt0_arm_android_argv+0x18(SB)/4,$0x2040  // HWCAP_VFP | HWCAP_VFPv3
DATA _rt0_arm_android_argv+0x1C(SB)/4,$0
GLOBL _rt0_arm_android_argv(SB),NOPTR,$0x20

DATA _rt0_arm_android_argv0(SB)/8, $"gojni"
GLOBL _rt0_arm_android_argv0(SB),RODATA,$8

DATA _rt0_arm_android_auxv0(SB)/4, $"v7l"
GLOBL _rt0_arm_android_auxv0(SB),RODATA,$4

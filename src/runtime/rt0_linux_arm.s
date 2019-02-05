// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_linux(SB),NOSPLIT|NOFRAME,$0
	MOVW	(R13), R0	// argc
	MOVW	$4(R13), R1		// argv
	MOVW	$_rt0_arm_linux1(SB), R4
	B		(R4)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm_linux_lib(SB),NOSPLIT,$0
	B	_rt0_arm_lib(SB)

TEXT _rt0_arm_linux1(SB),NOSPLIT|NOFRAME,$0
	// We first need to detect the kernel ABI, and warn the user
	// if the system only supports OABI.
	// The strategy here is to call some EABI syscall to see if
	// SIGILL is received.
	// If you get a SIGILL here, you have the wrong kernel.

	// Save argc and argv (syscall will clobber at least R0).
	MOVM.DB.W [R0-R1], (R13)

	// do an EABI syscall
	MOVW	$20, R7 // sys_getpid
	SWI	$0 // this will trigger SIGILL on OABI systems

	MOVM.IA.W (R13), [R0-R1]
	B	runtime·rt0_go(SB)

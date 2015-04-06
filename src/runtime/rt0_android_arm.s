// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_android(SB),NOSPLIT,$-4
	MOVW		(R13), R0      // argc
	MOVW		$4(R13), R1    // argv
	MOVW		$_rt0_arm_linux1(SB), R4
	B		(R4)

// This symbol is called when a shared library is loaded.
TEXT _rt0_arm_android_lib(SB),NOSPLIT,$0
	// TODO(crawshaw): initialize runtime.
	// At the moment this is done in mobile/app/android.c:init_go_runtime
	RET

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_darwin(SB),7,$-4
	// prepare arguments for main (_rt0_go)
	MOVW	(R13), R0	// argc
	MOVW	$4(R13), R1		// argv
	MOVW	$main(SB), R4
	B		(R4)

TEXT main(SB),NOSPLIT,$-8
	// save argc and argv onto stack
	MOVM.DB.W [R0-R1], (R13)
	MOVW	$runtime·rt0_go(SB), R4
	B		(R4)

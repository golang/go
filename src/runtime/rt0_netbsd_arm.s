// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm_netbsd(SB),NOSPLIT,$-4
	MOVW	(R13), R0		// argc
	MOVW	$4(R13), R1		// argv
	MOVM.DB.W [R0-R1], (R13)
	B runtime·rt0_go(SB)

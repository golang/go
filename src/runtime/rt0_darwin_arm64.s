// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// No need for _rt0_arm64_darwin as darwin/arm64 only
// supports external linking.
TEXT _rt0_arm64_darwin(SB),NOSPLIT,$-8
	MOVD	$42, R0
	MOVD	$1, R16	// SYS_exit
	SVC	$0x80

TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R2
	BL	(R2)
exit:
	MOVD	$0, R0
	MOVD	$1, R16	// sys_exit
	SVC	$0x80
	B	exit

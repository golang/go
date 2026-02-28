// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_arm64_netbsd(SB),NOSPLIT,$0
	MOVD	0(RSP), R0	// argc
	ADD	$8, RSP, R1	// argv
	JMP	main(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_netbsd_lib(SB),NOSPLIT,$0
	JMP	_rt0_arm64_lib(SB)

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// FreeBSD passes a pointer to the argument block in R0, not RSP,
// so _rt0_arm64 cannot be used.
TEXT _rt0_arm64_freebsd(SB),NOSPLIT,$0
	ADD	$8, R0, R1	// argv (use R0 while it's still the pointer)
	MOVD	0(R0), R0	// argc
	JMP	runtime·rt0_go(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_freebsd_lib(SB),NOSPLIT,$0
	JMP	_rt0_arm64_lib(SB)

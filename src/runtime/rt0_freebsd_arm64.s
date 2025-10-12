// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// On FreeBSD argc/argv are passed in R0, not RSP
TEXT _rt0_arm64_freebsd(SB),NOSPLIT,$0
	JMP	_rt0_arm64(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_freebsd_lib(SB),NOSPLIT,$0
	JMP	_rt0_arm64_lib(SB)

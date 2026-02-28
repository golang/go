// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// This is the entry point for the program from the
// kernel for an ordinary -buildmode=exe program.
TEXT _rt0_arm64_windows(SB),NOSPLIT,$0
	// Windows doesn't use argc and argv,
	// so there is no need to go through _rt0_arm64.
	JMP	runtime·rt0_go(SB)

// When building with -buildmode=c-shared, this symbol is called when the shared
// library is loaded.
TEXT _rt0_arm64_windows_lib(SB),NOSPLIT,$0
	// We get the argc and argv parameters from Win32.
	MOVD	$0, R0
	MOVD	$0, R1
	JMP	_rt0_arm64_lib(SB)

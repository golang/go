// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in R0.
// A single int32 result is returned in R0.
// (For more results, make an args/results structure.)
TEXT ·__system_property_get_trampoline(SB),NOSPLIT,$0
	MOVD	8(R0), R1		// arg 2 - value
	MOVD	0(R0), R0		// arg 1 - name
	CALL	libc___system_property_get(SB)
	RET

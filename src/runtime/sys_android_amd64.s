// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// These trampolines help convert from Go calling convention to C calling convention.
// They should be called with asmcgocall.
// A pointer to the arguments is passed in DI.
// A single int32 result is returned in AX.
// (For more results, make an args/results structure.)
TEXT runtime·__system_property_get_trampoline(SB),NOSPLIT,$0
	MOVQ	8(DI), SI		// arg 2 - value
	MOVQ	0(DI), DI		// arg 1 - name
	CALL	libc___system_property_get(SB)
	RET

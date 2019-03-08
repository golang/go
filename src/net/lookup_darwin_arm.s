// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM64, Darwin
// System calls are implemented in libSystem, this file contains
// trampolines that convert from Go to C calling convention.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT runtime·res_search_trampoline(SB),NOSPLIT,$0
    MOVW 16(R0). R4 // arg 5 anslen
    MOVW 12(R0), R3 // arg 4 answer
    MOVW 4(R0), R1  // arg 2 class
    MOVW 8(R0), R2  // arg 3 type 
    MOVW 0(R0), R0  // arg 1 name
    BL libc_res_search
    RET 
// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM64, Darwin
// System calls are implemented in libSystem, this file contains
// trampolines that convert from Go to C calling convention.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// On darwin/arm, the runtime always uses runtime/cgo
// for resolution. This will just exit with a nominal
// exit code.

TEXT runtime·res_search_trampoline(SB),NOSPLIT,$0
    MOVW    $90, R0
    BL    libc_exit(SB)
    RET

TEXT runtime·res_init_trampoline(SB),NOSPLIT,$0
    MOVW    $91, R0
    BL    libc_exit(SB)
    RET

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// System calls and other sys.stuff for ARM64, Darwin
// System calls are implemented in libSystem, this file contains
// trampolines that convert from Go to C calling convention.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

// On darwin/arm, the runtime always use runtime/cgo
// for resolution. This will just exit with nominal
// exit code

TEXT runtime·res_search_trampoline(SB),NOSPLIT,$0
    MOVW    $98, R0
    BL    libc_exit(SB)
    RET

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

#include "textflag.h"

TEXT ·libc_faccessat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_faccessat(SB)

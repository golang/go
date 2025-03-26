// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build openbsd && !mips64

#include "textflag.h"

TEXT ·libc_faccessat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_faccessat(SB)
TEXT ·libc_arc4random_buf_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_arc4random_buf(SB)
TEXT ·libc_readlinkat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_readlinkat(SB)
TEXT ·libc_mkdirat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_mkdirat(SB)
TEXT ·libc_fchmodat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_fchmodat(SB)
TEXT ·libc_fchownat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_fchownat(SB)
TEXT ·libc_renameat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_renameat(SB)
TEXT ·libc_linkat_trampoline(SB),NOSPLIT,$0-0
        JMP	libc_linkat(SB)

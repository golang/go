// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && arm64 && gc

#include "textflag.h"

TEXT libc_sysctlbyname_trampoline<>(SB),NOSPLIT,$0-0
	JMP	libc_sysctlbyname(SB)
GLOBL	·libc_sysctlbyname_trampoline_addr(SB), RODATA, $8
DATA	·libc_sysctlbyname_trampoline_addr(SB)/8, $libc_sysctlbyname_trampoline<>(SB)

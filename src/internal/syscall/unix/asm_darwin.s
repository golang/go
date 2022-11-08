// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·libc_getentropy_trampoline(SB),NOSPLIT,$0-0
	JMP	libc_getentropy(SB)

TEXT ·libc_getaddrinfo_trampoline(SB),NOSPLIT,$0-0
	JMP	libc_getaddrinfo(SB)

TEXT ·libc_freeaddrinfo_trampoline(SB),NOSPLIT,$0-0
	JMP	libc_freeaddrinfo(SB)

TEXT ·libc_getnameinfo_trampoline(SB),NOSPLIT,$0-0
	JMP	libc_getnameinfo(SB)

TEXT ·libc_gai_strerror_trampoline(SB),NOSPLIT,$0-0
	JMP	libc_gai_strerror(SB)

TEXT ·libresolv_res_9_ninit_trampoline(SB),NOSPLIT,$0-0
	JMP	libresolv_res_9_ninit(SB)

TEXT ·libresolv_res_9_nclose_trampoline(SB),NOSPLIT,$0-0
	JMP	libresolv_res_9_nclose(SB)

TEXT ·libresolv_res_9_nsearch_trampoline(SB),NOSPLIT,$0-0
	JMP	libresolv_res_9_nsearch(SB)

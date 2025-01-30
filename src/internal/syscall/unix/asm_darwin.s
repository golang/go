// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT ·libc_arc4random_buf_trampoline(SB),NOSPLIT,$0-0; JMP libc_arc4random_buf(SB)
TEXT ·libc_getaddrinfo_trampoline(SB),NOSPLIT,$0-0; JMP libc_getaddrinfo(SB)
TEXT ·libc_freeaddrinfo_trampoline(SB),NOSPLIT,$0-0; JMP libc_freeaddrinfo(SB)
TEXT ·libc_getnameinfo_trampoline(SB),NOSPLIT,$0-0; JMP libc_getnameinfo(SB)
TEXT ·libc_gai_strerror_trampoline(SB),NOSPLIT,$0-0; JMP libc_gai_strerror(SB)
TEXT ·libresolv_res_9_ninit_trampoline(SB),NOSPLIT,$0-0; JMP libresolv_res_9_ninit(SB)
TEXT ·libresolv_res_9_nclose_trampoline(SB),NOSPLIT,$0-0; JMP libresolv_res_9_nclose(SB)
TEXT ·libresolv_res_9_nsearch_trampoline(SB),NOSPLIT,$0-0; JMP libresolv_res_9_nsearch(SB)
TEXT ·libc_grantpt_trampoline(SB),NOSPLIT,$0-0; JMP libc_grantpt(SB)
TEXT ·libc_unlockpt_trampoline(SB),NOSPLIT,$0-0; JMP libc_unlockpt(SB)
TEXT ·libc_ptsname_r_trampoline(SB),NOSPLIT,$0-0; JMP libc_ptsname_r(SB)
TEXT ·libc_posix_openpt_trampoline(SB),NOSPLIT,$0-0; JMP libc_posix_openpt(SB)
TEXT ·libc_getgrouplist_trampoline(SB),NOSPLIT,$0-0; JMP libc_getgrouplist(SB)
TEXT ·libc_getpwnam_r_trampoline(SB),NOSPLIT,$0-0; JMP libc_getpwnam_r(SB)
TEXT ·libc_getpwuid_r_trampoline(SB),NOSPLIT,$0-0; JMP libc_getpwuid_r(SB)
TEXT ·libc_getgrnam_r_trampoline(SB),NOSPLIT,$0-0; JMP libc_getgrnam_r(SB)
TEXT ·libc_getgrgid_r_trampoline(SB),NOSPLIT,$0-0; JMP libc_getgrgid_r(SB)
TEXT ·libc_sysconf_trampoline(SB),NOSPLIT,$0-0; JMP libc_sysconf(SB)
TEXT ·libc_faccessat_trampoline(SB),NOSPLIT,$0-0; JMP libc_faccessat(SB)
TEXT ·libc_readlinkat_trampoline(SB),NOSPLIT,$0-0; JMP libc_readlinkat(SB)
TEXT ·libc_mkdirat_trampoline(SB),NOSPLIT,$0-0; JMP libc_mkdirat(SB)
TEXT ·libc_fchmodat_trampoline(SB),NOSPLIT,$0-0; JMP libc_fchmodat(SB)

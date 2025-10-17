// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT syscall_check0_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check0(SB)
GLOBL   ·syscall_check0_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check0_trampoline_addr(SB)/8, $syscall_check0_trampoline<>(SB)

TEXT syscall_check1_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check1(SB)
GLOBL   ·syscall_check1_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check1_trampoline_addr(SB)/8, $syscall_check1_trampoline<>(SB)

TEXT syscall_check2_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check2(SB)
GLOBL   ·syscall_check2_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check2_trampoline_addr(SB)/8, $syscall_check2_trampoline<>(SB)

TEXT syscall_check3_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check3(SB)
GLOBL   ·syscall_check3_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check3_trampoline_addr(SB)/8, $syscall_check3_trampoline<>(SB)

TEXT syscall_check4_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check4(SB)
GLOBL   ·syscall_check4_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check4_trampoline_addr(SB)/8, $syscall_check4_trampoline<>(SB)

TEXT syscall_check5_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check5(SB)
GLOBL   ·syscall_check5_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check5_trampoline_addr(SB)/8, $syscall_check5_trampoline<>(SB)

TEXT syscall_check6_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check6(SB)
GLOBL   ·syscall_check6_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check6_trampoline_addr(SB)/8, $syscall_check6_trampoline<>(SB)

TEXT syscall_check7_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check7(SB)
GLOBL   ·syscall_check7_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check7_trampoline_addr(SB)/8, $syscall_check7_trampoline<>(SB)

TEXT syscall_check8_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check8(SB)
GLOBL   ·syscall_check8_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check8_trampoline_addr(SB)/8, $syscall_check8_trampoline<>(SB)

TEXT syscall_check9_trampoline<>(SB),NOSPLIT,$0-0
    JMP syscall_check9(SB)
GLOBL   ·syscall_check9_trampoline_addr(SB), RODATA, $8
DATA    ·syscall_check9_trampoline_addr(SB)/8, $syscall_check9_trampoline<>(SB)

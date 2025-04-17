// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#define RODATA	8

DATA ·smallIntAsm(SB)/8, $const_smallInt
GLOBL ·smallIntAsm(SB),RODATA,$8

DATA ·bigIntAsm(SB)/8, $const_bigInt
GLOBL ·bigIntAsm(SB),RODATA,$8

DATA ·stringAsm(SB)/4, $const_stringVal
GLOBL ·stringAsm(SB),RODATA,$4

DATA ·longStringAsm(SB)/91, $const_longStringVal
GLOBL ·longStringAsm(SB),RODATA,$91

DATA ·typSize(SB)/8, $typ__size
GLOBL ·typSize(SB),RODATA,$8

DATA ·typA(SB)/8, $typ_a
GLOBL ·typA(SB),RODATA,$8

DATA ·typB(SB)/8, $typ_b
GLOBL ·typB(SB),RODATA,$8

DATA ·typC(SB)/8, $typ_c
GLOBL ·typC(SB),RODATA,$8

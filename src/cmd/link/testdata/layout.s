// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of section assignment in layout.go.
// Each symbol should end up in the section named by the symbol name prefix (up to the underscore).

#include "../../ld/textflag.h"

TEXT text_start(SB),7,$0
	MOVQ $rodata_sym(SB), AX
	MOVQ $noptrdata_sym(SB), AX
	MOVQ $data_sym(SB), AX
	MOVQ $bss_sym(SB), AX
	MOVQ $noptrbss_sym(SB), AX
	RET

DATA rodata_sym(SB)/4, $1
GLOBL rodata_sym(SB), RODATA, $4

DATA noptrdata_sym(SB)/4, $1
GLOBL noptrdata_sym(SB), NOPTR, $4

DATA data_sym(SB)/4, $1
GLOBL data_sym(SB), $4

GLOBL bss_sym(SB), $4

GLOBL noptrbss_sym(SB), NOPTR, $4

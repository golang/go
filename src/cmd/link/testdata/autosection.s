// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of section-named symbols.

#include "../../ld/textflag.h"

TEXT start(SB),7,$0
	MOVQ $autotab(SB),AX
	MOVQ $autoptr(SB),AX
	RET

GLOBL zero(SB), $8

GLOBL zeronoptr(SB), NOPTR, $16

// text
DATA autotab+0x00(SB)/8, $text(SB)
DATA autotab+0x08(SB)/8, $start(SB)
DATA autotab+0x10(SB)/8, $etext(SB)
DATA autotab+0x18(SB)/8, $start+16(SB)

// data
DATA autotab+0x20(SB)/8, $data(SB)
DATA autotab+0x28(SB)/8, $autotab(SB)
DATA autotab+0x30(SB)/8, $edata(SB)
DATA autotab+0x38(SB)/8, $nonzero+4(SB)

// bss
DATA autotab+0x40(SB)/8, $bss(SB)
DATA autotab+0x48(SB)/8, $zero(SB)
DATA autotab+0x50(SB)/8, $ebss(SB)
DATA autotab+0x58(SB)/8, $zero+8(SB)

// noptrdata
DATA autotab+0x60(SB)/8, $noptrdata(SB)
DATA autotab+0x68(SB)/8, $nonzeronoptr(SB)
DATA autotab+0x70(SB)/8, $enoptrdata(SB)
DATA autotab+0x78(SB)/8, $nonzeronoptr+8(SB)

// noptrbss
DATA autotab+0x80(SB)/8, $noptrbss(SB)
DATA autotab+0x88(SB)/8, $zeronoptr(SB)
DATA autotab+0x90(SB)/8, $enoptrbss(SB)
DATA autotab+0x98(SB)/8, $zeronoptr+16(SB)

// end
DATA autotab+0xa0(SB)/8, $end(SB)
DATA autotab+0xa8(SB)/8, $zeronoptr+16(SB)

GLOBL autotab(SB), $0xb0

DATA nonzero(SB)/4, $1
GLOBL nonzero(SB), $4

DATA nonzeronoptr(SB)/8, $2
GLOBL nonzeronoptr(SB), NOPTR, $8

GLOBL autoptr(SB), $0

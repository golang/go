// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of dead code removal.
// Symbols with names beginning with dead_ should be discarded.
// Others should be kept.

TEXT start(SB),7,$0	// start symbol
	MOVQ $data1<>(SB), AX
	CALL text1(SB)
	MOVQ $text2(SB), BX
	RET

TEXT text1(SB),7,$0
	FUNCDATA $1, funcdata+4(SB)
	RET

TEXT text2(SB),7,$0
	MOVQ $runtime·edata(SB),BX
	RET

DATA data1<>+0(SB)/8, $data2(SB)
DATA data1<>+8(SB)/8, $data3(SB)
GLOBL data1<>(SB), $16
GLOBL data2(SB), $1
GLOBL data3(SB), $1
GLOBL funcdata(SB), $8

TEXT dead_start(SB),7,$0
	MOVQ $dead_data1(SB), AX
	CALL dead_text1(SB)
	MOVQ $dead_text2(SB), BX
	RET

TEXT dead_text1(SB),7,$0
	FUNCDATA $1, dead_funcdata+4(SB)
	RET

TEXT dead_text2(SB),7,$0
	RET

DATA dead_data1+0(SB)/8, $dead_data2(SB)
DATA dead_data1+8(SB)/8, $dead_data3(SB)
GLOBL dead_data1(SB), $16
GLOBL dead_data2(SB), $1
GLOBL dead_data3(SB), $1
GLOBL dead_funcdata(SB), $8

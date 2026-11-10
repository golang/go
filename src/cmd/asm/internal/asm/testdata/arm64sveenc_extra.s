// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This input extends auto-generated arm64sveenc.s test suite
// with manually added tests.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$-8
	// Byte-scaled SP-relative operands are materialized through REGTMP:
	//	ZSTR Z0, 40(RSP) -> MOVD $40(RSP), R27; ZSTR Z0, (VL*0)(R27)
	// Only the first encoding on a line is checked.
	ZSTR Z0, 40(RSP)                   // fba30091
	ZSTR Z0, (VL*0)(R27)               // 604380e5
	ZLDR 40(RSP), Z2                   // fba30091
	ZLDR (VL*0)(R27), Z2               // 62438085
	PSTR P1, 16(RSP)                   // fb430091
	PSTR P1, (VL*0)(R27)               // 610380e5
	PLDR -8(RSP), P0                   // fb2300d1
	PLDR (VL*0)(R27), P0               // 60038085
	// 8-bit narrowing shifts of two .H registers: imm3 holds 8 - shift. The
	// generated suite skips these because binutils 2.45 does not know them.
	ZSQRSHRN $6, [Z2.H-Z3.H], Z10.B    // 4a28aa45
	ZSQRSHRUN $6, [Z2.H-Z3.H], Z10.B   // 4a08aa45
	ZUQRSHRN $6, [Z2.H-Z3.H], Z10.B    // 4a38aa45
	ZSQRSHRN $1, [Z2.H-Z3.H], Z10.B    // 4a28af45
	ZSQRSHRN $8, [Z2.H-Z3.H], Z10.B    // 4a28a845
	RET

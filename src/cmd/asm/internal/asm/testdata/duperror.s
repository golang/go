// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT foo(SB), 0, $0
	RET
TEXT foo(SB), 0, $0 // ERROR "symbol foo redeclared"
	RET

GLOBL bar(SB), 0, $8
GLOBL bar(SB), 0, $8 // ERROR "symbol bar redeclared"

DATA bar+0(SB)/8, $0
DATA bar+0(SB)/8, $0 // ERROR "overlapping DATA entry for bar"

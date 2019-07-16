// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·f(SB),0,$0-8
	MOVW	x+0(FP), R1
	MOVW	x+4(FP), R2
	DIVU	R1, R2
	DIV	R1, R2
	MODU	R1, R2
	MOD	R1, R2
	RET

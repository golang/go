// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	MULS.S	R1, R2, R3, R4     // ERROR "invalid .S suffix"
	ADD.P	R1, R2, R3         // ERROR "invalid .P suffix"
	SUB.W	R2, R3             // ERROR "invalid .W suffix"
	BL	4(R4)              // ERROR "non-zero offset"
	ADDF	F0, R1, F2         // ERROR "illegal combination"
	SWI	(R0)               // ERROR "illegal combination"
	NEGF	F0, F1, F2         // ERROR "illegal combination"
	NEGD	F0, F1, F2         // ERROR "illegal combination"
	ABSF	F0, F1, F2         // ERROR "illegal combination"
	ABSD	F0, F1, F2         // ERROR "illegal combination"
	SQRTF	F0, F1, F2         // ERROR "illegal combination"
	SQRTD	F0, F1, F2         // ERROR "illegal combination"
	MOVF	F0, F1, F2         // ERROR "illegal combination"
	MOVD	F0, F1, F2         // ERROR "illegal combination"
	MOVDF	F0, F1, F2         // ERROR "illegal combination"
	MOVFD	F0, F1, F2         // ERROR "illegal combination"

	END

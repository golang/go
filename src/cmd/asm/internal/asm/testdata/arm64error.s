// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	MOVD.P	300(R2), R3            // ERROR "offset out of range [-255,254]"
	MOVD.P	R3, 344(R2)            // ERROR "offset out of range [-255,254]"
	VLD1	(R8)(R13), [V2.B16]    // ERROR "illegal combination"
	VLD1	8(R9), [V2.B16]        // ERROR "illegal combination"
	VST1	[V1.B16], (R8)(R13)    // ERROR "illegal combination"
	VST1	[V1.B16], 9(R2)        // ERROR "illegal combination"
	VLD1	8(R8)(R13), [V2.B16]   // ERROR "illegal combination"
	ADD	R1.UXTB<<5, R2, R3     // ERROR "shift amount out of range 0 to 4"
	ADDS	R1.UXTX<<7, R2, R3     // ERROR "shift amount out of range 0 to 4"
	RET

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	MULS.S	R1, R2, R3, R4     // ERROR "invalid .S suffix"
	ADD.P	R1, R2, R3         // ERROR "invalid .P suffix"
	SUB.W	R2, R3             // ERROR "invalid .W suffix"
	BL	4(R4)              // ERROR "non-zero offset"
	END

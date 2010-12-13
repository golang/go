// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin and Linux use the same linkage to main

TEXT _rt0_386_linux(SB),7,$0
	// Linux starts the FPU in extended double precision.
	// Other operating systems use double precision.
	// Change to double precision to match them,
	// and to match other hardware that only has double.
	PUSHL $0x27F
	FLDCW	0(SP)
	POPL AX

	JMP	_rt0_386(SB)


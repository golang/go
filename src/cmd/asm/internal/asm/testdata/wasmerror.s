// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	BrTable	$0				// ERROR "BrTable is not supported in assembly"
	BrTable	$0, $1				// ERROR "BrTable is not supported in assembly"
	I64Load	R0				// ERROR "bad type for *Load"
	I64Load	$-8				// ERROR "bad offset"
	I64Load	x-16(FP)			// ERROR "bad offset"
	I64Load	$0x100000000			// ERROR "bad offset"
	I64Store	$-8			// ERROR "bad offset"
	I64Store	$0x100000000		// ERROR "bad offset"
	Block
	CALLNORESUME	foo(SB)			// no error
	CALL	foo(SB)				// ERROR "CALL can only be used on toplevel, try CALLNORESUME instead"
	RESUMEPOINT				// ERROR "RESUMEPOINT can only be used on toplevel"
	End
	End					// ERROR "End without block"
	RET

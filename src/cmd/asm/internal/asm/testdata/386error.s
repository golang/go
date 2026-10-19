// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	// A global symbol may be indexed, but it cannot have a base register.
	MOVL	foo<>(SB)(AX*4), AX	// no error
	MOVL	foo<>(SB)(AX*1), AX	// no error
	MOVL	AX, foo+4(SB)(DX*1)	// no error
	LEAL	foo<>(SB)(AX*1), AX	// no error
	MOVL	foo<>(SB)(AX), AX	// ERROR "base register not allowed with global symbol"
	MOVL	foo(SB)(AX), AX		// ERROR "base register not allowed with global symbol"
	MOVL	AX, foo+4(SB)(CX)	// ERROR "base register not allowed with global symbol"
	MOVL	foo<>(SB)(BX)(AX*4), AX	// ERROR "base register not allowed with global symbol"
	LEAL	foo<>(SB)(AX), AX	// ERROR "base register not allowed with global symbol"
	MOVL	$foo<>(SB)(AX), AX	// ERROR "base register not allowed with global symbol"
	// The address of a global symbol cannot be indexed.
	MOVL	$foo<>(SB)(AX*4), BX	// ERROR "index register not allowed with address of global symbol"
	RET

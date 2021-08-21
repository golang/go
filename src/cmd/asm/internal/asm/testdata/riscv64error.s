// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	MOV	$errors(SB), (X5)		// ERROR "unsupported addr MOV"
	MOV	$8(SP), (X5)			// ERROR "unsupported addr MOV"
	MOVB	$8(SP), X5			// ERROR "unsupported addr MOV"
	MOVH	$8(SP), X5			// ERROR "unsupported addr MOV"
	MOVW	$8(SP), X5			// ERROR "unsupported addr MOV"
	MOVF	$8(SP), X5			// ERROR "unsupported addr MOV"
	MOV	$1234, 0(SP)			// ERROR "constant load must target register"
	MOV	$1234, 8(SP)			// ERROR "constant load must target register"
	MOV	$0, 0(SP)			// ERROR "constant load must target register"
	MOV	$0, 8(SP)			// ERROR "constant load must target register"
	MOV	$1234, 0(SP)			// ERROR "constant load must target register"
	MOV	$1234, 8(SP)			// ERROR "constant load must target register"
	MOVB	$1, X5				// ERROR "unsupported constant load"
	MOVH	$1, X5				// ERROR "unsupported constant load"
	MOVW	$1, X5				// ERROR "unsupported constant load"
	MOVF	$1, X5				// ERROR "unsupported constant load"
	MOVBU	X5, (X6)			// ERROR "unsupported unsigned store"
	MOVHU	X5, (X6)			// ERROR "unsupported unsigned store"
	MOVWU	X5, (X6)			// ERROR "unsupported unsigned store"

	RET

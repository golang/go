// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	MOVL	foo<>(SB)(AX), AX	// ERROR "invalid instruction"
	MOVL	(AX)(SP*1), AX		// ERROR "invalid instruction"
	RET

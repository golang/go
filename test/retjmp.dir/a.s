// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT	·f(SB), 4, $8-0
	CALL	·f1(SB)
	RET	·f2(SB)
	CALL	·unreachable(SB)

TEXT	·leaf(SB), 4, $0-0
	RET	·f3(SB)
	JMP	·unreachable(SB)

TEXT	·leaf2(SB), 4, $32-0 // nonzero frame size
	RET	·f4(SB)
	JMP	·unreachable(SB)

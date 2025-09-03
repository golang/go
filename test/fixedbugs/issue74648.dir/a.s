// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT ·F(SB), $0
	JMP prealigned
	INT $3 // should never be reached
prealigned:
	PCALIGN $0x10
aligned:
	RET

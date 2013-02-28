// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build race

// func runtime·racefuncenter(pc uintptr)
TEXT	runtime·racefuncenter(SB), 7, $16
	MOVQ	DX, saved-8(SP) // save function entry context (for closures)
	MOVQ	pc+0(FP), DX
	MOVQ	DX, arg-16(SP)
	CALL	runtime·racefuncenter1(SB)
	MOVQ	saved-8(SP), DX
	RET

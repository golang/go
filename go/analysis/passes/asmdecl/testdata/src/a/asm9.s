// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build wasm

TEXT ·returnint(SB),NOSPLIT|NOFRAME,$0-8
	MOVD	24(SP), R1 // ok to access beyond stack frame with NOFRAME
	CallImport // interpreted as writing result
	RET

TEXT ·returnbyte(SB),$0-9
	RET // want `RET without writing`

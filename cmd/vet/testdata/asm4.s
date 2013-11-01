// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64
// +build vet_test

// Test cases for symbolic NOSPLIT etc. on TEXT symbols.

TEXT ·noprof(SB),NOPROF,$0-8
	RET

TEXT ·dupok(SB),DUPOK,$0-8
	RET

TEXT ·nosplit(SB),NOSPLIT,$0
	RET

TEXT ·rodata(SB),RODATA,$0-8
	RET

TEXT ·noptr(SB),NOPTR|NOSPLIT,$0
	RET

TEXT ·wrapper(SB),WRAPPER,$0-8
	RET

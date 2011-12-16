// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Darwin and Linux use the same linkage to main

TEXT _rt0_386_linux(SB),7,$0
	CALL	runtime·linux_setup_vdso(SB)
	JMP		_rt0_386(SB)

TEXT _fallback_vdso(SB),7,$0
	INT	$0x80
	RET

DATA	runtime·_vdso(SB)/4, $_fallback_vdso(SB)
GLOBL	runtime·_vdso(SB), $4


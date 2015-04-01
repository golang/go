// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_amd64_linux(SB),NOSPLIT,$-8
	LEAQ	8(SP), SI // argv
	MOVQ	0(SP), DI // argc
	MOVQ	$main(SB), AX
	JMP	AX

// When linking with -shared, this symbol is called when the shared library
// is loaded.
TEXT _rt0_amd64_linux_lib(SB),NOSPLIT,$0
	// TODO(spetrovic): Do something useful, like calling $main.  (Note that
	// this has to be done in a separate thread, as main is expected to block.)
	RET

TEXT main(SB),NOSPLIT,$-8
	MOVQ	$runtime·rt0_go(SB), AX
	JMP	AX

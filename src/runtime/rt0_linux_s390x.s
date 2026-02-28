// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

TEXT _rt0_s390x_linux(SB), NOSPLIT|NOFRAME, $0
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.

	MOVD 0(R15), R2  // argc
	ADD  $8, R15, R3 // argv
	BR   main(SB)

TEXT _rt0_s390x_linux_lib(SB), NOSPLIT, $0
	MOVD $_rt0_s390x_lib(SB), R1
	BR   R1

TEXT main(SB), NOSPLIT|NOFRAME, $0
	MOVD $runtime·rt0_go(SB), R1
	BR   R1

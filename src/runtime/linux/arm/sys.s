// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//
// System calls and other sys.stuff for arm, Linux
//

TEXT write(SB),7,$0
	MOVW	4(SP), R0
	MOVW	8(SP), R1
	MOVW	12(SP), R2
    	SWI	$0x00900004  // syscall write
	RET


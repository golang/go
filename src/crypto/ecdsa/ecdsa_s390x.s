// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func kdsaSig(fc uint64, block *[1720]byte) (errn uint64)
TEXT ·kdsaSig(SB), NOSPLIT|NOFRAME, $0-24
	MOVD fc+0(FP), R0    // function code
	MOVD block+8(FP), R1 // address parameter block

loop:
	WORD $0xB93A0008 // compute digital signature authentication
	BVS  loop        // branch back if interrupted
	BEQ  success     // signature creation successful
	BGT  retry       // signing unsuccessful, but retry with new CSPRN

error:
	MOVD $2, R2          // fallthrough indicates fatal error
	MOVD R2, errn+16(FP) // return 2 - sign/verify abort
	RET

retry:
	MOVD $1, R2
	MOVD R2, errn+16(FP) // return 1 - sign/verify was unsuccessful -- if sign, retry with new RN
	RET

success:
	MOVD $0, R2
	MOVD R2, errn+16(FP) // return 0 - sign/verify was successful
	RET

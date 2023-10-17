// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// func kdsa(fc uint64, params *[4096]byte) (errn uint64)
TEXT ·kdsa(SB), NOSPLIT|NOFRAME, $0-24
	MOVD fc+0(FP), R0     // function code
	MOVD params+8(FP), R1 // address parameter block

loop:
	KDSA R0, R4      // compute digital signature authentication
	BVS  loop        // branch back if interrupted
	BGT  retry       // signing unsuccessful, but retry with new CSPRN
	BLT  error       // condition code of 1 indicates a failure

success:
	MOVD $0, errn+16(FP) // return 0 - sign/verify was successful
	RET

error:
	MOVD $1, errn+16(FP) // return 1 - sign/verify failed
	RET

retry:
	MOVD $2, errn+16(FP) // return 2 - sign/verify was unsuccessful -- if sign, retry with new RN
	RET

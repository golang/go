// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0
	// Instructions that were encoded with BYTE sequences.
	// Included to simplify validation of CL that fixed that.
	MOVQ (AX), M0  // 0f6f00
	MOVQ M0, 8(SP) // 0f7f442408
	MOVQ 8(SP), M0 // 0f6f442408
	MOVQ M0, (AX)  // 0f7f00
	MOVQ M0, (BX)  // 0f7f03
	// End of tests.
	RET

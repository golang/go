// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "funcdata.h"
#include "textflag.h"

// Assembly function for runtime_test.TestStartLineAsm.
//
// Note that this file can't be built directly as part of runtime_test, as assembly
// files can't declare an alternative package. Building it into runtime is
// possible, but linkshared complicates things:
//
//  1. linkshared mode leaves the function around in the final output of
//     non-test builds.
//  2. Due of (1), the linker can't resolve the callerStartLine relocation
//     (as runtime_test isn't built for non-test builds).
//
// Thus it is simpler to just put this in its own package, imported only by
// runtime_test. We use ABIInternal as no ABI wrapper is generated for
// callerStartLine since it is in a different package.

TEXT	·AsmFunc<ABIInternal>(SB),NOSPLIT,$8-0
	NO_LOCAL_POINTERS
	MOVQ	$0, AX // wantInlined
	MOVQ	·CallerStartLine(SB), DX
	CALL	(DX)
	RET

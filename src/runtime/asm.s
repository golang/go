// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

#ifndef GOARCH_amd64
TEXT ·sigpanic0(SB),NOSPLIT,$0-0
	JMP	·sigpanic<ABIInternal>(SB)
#endif

// See map.go comment on the need for this routine.
TEXT ·mapinitnoop<ABIInternal>(SB),NOSPLIT,$0-0
	RET

#ifndef GOARCH_amd64
#ifndef GOARCH_arm64
#ifndef GOARCH_loong64
#ifndef GOARCH_mips64
#ifndef GOARCH_mips64le
#ifndef GOARCH_ppc64
#ifndef GOARCH_ppc64le
#ifndef GOARCH_riscv64
#ifndef GOARCH_s390x
#ifndef GOARCH_wasm
// stub to appease shared build mode.
TEXT ·switchToCrashStack0<ABIInternal>(SB),NOSPLIT,$0-0
	UNDEF
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif
#endif

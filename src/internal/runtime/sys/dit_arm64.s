// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT ·EnableDIT(SB),$0-1
    MRS DIT, R0
    UBFX $24, R0, $1, R1
    MOVB R1, ret+0(FP)
    TBNZ $0, R1, ret
    MSR $1, DIT
#ifdef GOOS_darwin
    // Arm documents that barriers are not necessary when writing to, or reading
    // from, PSTATE fields. However, Apple documentation indicates that barriers
    // should be used, in particular when setting the PSTATE.DIT field. Barriers
    // aren't cheap, so only use them on Apple silicon for now.
    //
    // See go.dev/issue/77776.
    MOVBU internal∕cpu·ARM64+const_offsetARM64HasSB(SB), R2
    TBZ $0, R2, sbFallback
    SB
#endif
ret:
    RET
sbFallback:
    DSB $7  // nsh
    ISB $15 // sy
    RET

TEXT ·DITEnabled(SB),$0-1
    MRS DIT, R0
    UBFX $24, R0, $1, R1
    MOVB R1, ret+0(FP)
    RET

TEXT ·DisableDIT(SB),$0
    MSR $0, DIT
    RET

#include "textflag.h"

TEXT ·EnableDIT(SB),$0-1
    MRS DIT, R0
    UBFX $24, R0, $1, R1
    MOVB R1, ret+0(FP)
    MSR $1, DIT
    RET

TEXT ·DITEnabled(SB),$0-1
    MRS DIT, R0
    UBFX $24, R0, $1, R1
    MOVB R1, ret+0(FP)
    RET

TEXT ·DisableDIT(SB),$0
    MSR $0, DIT
    RET

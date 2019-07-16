#include "textflag.h"

DATA ·pointer(SB)/4, $·target(SB)
GLOBL ·pointer(SB),RODATA,$4

TEXT ·jump(SB),NOSPLIT,$4
        CALL *·pointer(SB)
        RET

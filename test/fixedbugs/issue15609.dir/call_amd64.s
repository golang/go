#include "textflag.h"

DATA ·pointer(SB)/8, $·target(SB)
GLOBL ·pointer(SB),RODATA,$8

TEXT ·jump(SB),NOSPLIT,$8
        CALL *·pointer(SB)
        RET

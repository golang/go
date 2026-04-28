// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"


TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
	MOVD    R5, R4
	AND	$0xff, R4
	BR      indexbytebody<>(SB)

TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-32
	AND	$0xff, R4
	BR      indexbytebody<>(SB)

// input:
// R2: s
// R3: s_len
// R4: c -- byte sought
// For regabiargs output value(index) stored in R2
// For !regabiargs address of output value(index) stored in R5
TEXT indexbytebody<>(SB),NOSPLIT|NOFRAME,$0
        CMPBEQ  R3, $0, notfound
        MOVD    R2, R6          // store base for later
        ADD     R2, R3, R8      // the address after the end of the string
        //if the length is small, use loop; otherwise, use vector or srst search
        CMPBGE  R3, $16, large

residual:
        CMPBEQ  R2, R8, notfound
        MOVBZ   0(R2), R7
        LA      1(R2), R2
        CMPBNE  R7, R4, residual

found:
        SUB     R6, R2
        SUB     $1, R2
        RET

notfound:
        MOVD    $-1, R2
        RET

large:
        MOVBZ   internal∕cpu·S390X+const_offsetS390xHasVX(SB), R1
        CMPBNE  R1, $0, vectorimpl

srstimpl:                       // no vector facility
        MOVBZ   R4, R0          // c needs to be in R0, leave until last minute as currently R0 is expected to be 0
srstloop:
        WORD    $0xB25E0082     // srst %r8, %r2 (search the range [R2, R8))
        BVS     srstloop        // interrupted - continue
        BGT     notfoundr0
foundr0:
        XOR     R0, R0          // reset R0
        SUB     R6, R8          // remove base
        MOVD    R8, R2
        RET
notfoundr0:
        XOR     R0, R0          // reset R0
        MOVD    $-1, R2
        RET

vectorimpl:
        //if the address is not 16byte aligned, use loop for the header
        MOVD    R2, R8
        AND     $15, R8
        CMPBGT  R8, $0, notaligned

aligned:
        ADD     R6, R3, R8
        MOVD    R8, R7
        AND     $-16, R7
        // replicate c across V17
        VLVGB   $0, R4, V19
        VREPB   $0, V19, V17

vectorloop:
        CMPBGE  R2, R7, residual
        VL      0(R2), V16    // load string to be searched into V16
        ADD     $16, R2
        VFEEBS  V16, V17, V18 // search V17 in V16 and set conditional code accordingly
        BVS     vectorloop

        // when vector search found c in the string
        VLGVB   $7, V18, R7   // load 7th element of V18 containing index into R7
        SUB     $16, R2
        SUB     R6, R2
        ADD     R2, R7
        MOVD    R7, R2
        RET

notaligned:
        MOVD    R2, R8
        AND     $-16, R8
        ADD     $16, R8
notalignedloop:
        CMPBEQ  R2, R8, aligned
        MOVBZ   0(R2), R7
        LA      1(R2), R2
        CMPBNE  R7, R4, notalignedloop
        BR      found


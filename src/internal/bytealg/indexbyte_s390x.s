// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"


TEXT ·IndexByte<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
#ifndef GOEXPERIMENT_regabiargs
	MOVD    b_base+0(FP), R2// b_base => R2
	MOVD    b_len+8(FP), R3 // b_len => R3
	MOVBZ   c+24(FP), R4    // c => R4
	MOVD    $ret+32(FP), R5 // &ret => R5
#else
	MOVD    R5, R4
	AND	$0xff, R4
#endif
	BR      indexbytebody<>(SB)

TEXT ·IndexByteString<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-32
#ifndef GOEXPERIMENT_regabiargs
	MOVD    s_base+0(FP), R2 // s_base => R2
	MOVD    s_len+8(FP), R3 // s_len => R3
	MOVBZ   c+16(FP), R4    // c => R4
	MOVD    $ret+24(FP), R5 // &ret => R5
#else
	AND	$0xff, R4
#endif
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
#ifndef GOEXPERIMENT_regabiargs
        MOVD    R2, 0(R5)
#endif
        RET

notfound:
#ifndef GOEXPERIMENT_regabiargs
        MOVD    $-1, 0(R5)
#else
        MOVD    $-1, R2
#endif
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
#ifndef GOEXPERIMENT_regabiargs
        MOVD    R8, 0(R5)
#else
        MOVD    R8, R2
#endif
        RET
notfoundr0:
        XOR     R0, R0          // reset R0
#ifndef GOEXPERIMENT_regabiargs
        MOVD    $-1, 0(R5)
#else
        MOVD    $-1, R2
#endif
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
#ifndef GOEXPERIMENT_regabiargs
        MOVD    R7, 0(R5)
#else
        MOVD    R7, R2
#endif
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


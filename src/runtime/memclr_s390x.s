// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers(SB),NOSPLIT|NOFRAME,$0-16
	MOVD	ptr+0(FP), R4
	MOVD	n+8(FP), R5

start:
	CMPBLE	R5, $3, clear0to3
	CMPBLE	R5, $7, clear4to7
	CMPBLE	R5, $11, clear8to11
	CMPBLE	R5, $15, clear12to15
	CMP	R5, $32
	BGE	clearmt32
	MOVD	$0, 0(R4)
	MOVD	$0, 8(R4)
	ADD	$16, R4
	SUB	$16, R5
	BR	start

clear0to3:
	CMPBEQ	R5, $0, done
	CMPBNE	R5, $1, clear2
	MOVB	$0, 0(R4)
	RET
clear2:
	CMPBNE	R5, $2, clear3
	MOVH	$0, 0(R4)
	RET
clear3:
	MOVH	$0, 0(R4)
	MOVB	$0, 2(R4)
	RET

clear4to7:
	CMPBNE	R5, $4, clear5
	MOVW	$0, 0(R4)
	RET
clear5:
	CMPBNE	R5, $5, clear6
	MOVW	$0, 0(R4)
	MOVB	$0, 4(R4)
	RET
clear6:
	CMPBNE	R5, $6, clear7
	MOVW	$0, 0(R4)
	MOVH	$0, 4(R4)
	RET
clear7:
	MOVW	$0, 0(R4)
	MOVH	$0, 4(R4)
	MOVB	$0, 6(R4)
	RET

clear8to11:
	CMPBNE	R5, $8, clear9
	MOVD	$0, 0(R4)
	RET
clear9:
	CMPBNE	R5, $9, clear10
	MOVD	$0, 0(R4)
	MOVB	$0, 8(R4)
	RET
clear10:
	CMPBNE	R5, $10, clear11
	MOVD	$0, 0(R4)
	MOVH	$0, 8(R4)
	RET
clear11:
	MOVD	$0, 0(R4)
	MOVH	$0, 8(R4)
	MOVB	$0, 10(R4)
	RET

clear12to15:
	CMPBNE	R5, $12, clear13
	MOVD	$0, 0(R4)
	MOVW	$0, 8(R4)
	RET
clear13:
	CMPBNE	R5, $13, clear14
	MOVD	$0, 0(R4)
	MOVW	$0, 8(R4)
	MOVB	$0, 12(R4)
	RET
clear14:
	CMPBNE	R5, $14, clear15
	MOVD	$0, 0(R4)
	MOVW	$0, 8(R4)
	MOVH	$0, 12(R4)
	RET
clear15:
	MOVD	$0, 0(R4)
	MOVW	$0, 8(R4)
	MOVH	$0, 12(R4)
	MOVB	$0, 14(R4)
	RET

clearmt32:
	CMP	R5, $256
	BLT	clearlt256
	XC	$256, 0(R4), 0(R4)
	ADD	$256, R4
	ADD	$-256, R5
	BR	clearmt32
clearlt256:
	CMPBEQ	R5, $0, done
	ADD	$-1, R5
	EXRL	$memclr_exrl_xc<>(SB), R5
done:
	RET

// DO NOT CALL - target for exrl (execute relative long) instruction.
TEXT memclr_exrl_xc<>(SB),NOSPLIT|NOFRAME,$0-0
	XC	$1, 0(R4), 0(R4)
	MOVD	$0, 0(R0)
	RET


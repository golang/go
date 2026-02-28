// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// See memclrNoHeapPointers Go doc for important implementation constraints.

// func memclrNoHeapPointers(ptr unsafe.Pointer, n uintptr)
TEXT runtime·memclrNoHeapPointers<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-16
#ifndef GOEXPERIMENT_regabiargs
	MOVD	ptr+0(FP), R4
	MOVD	n+8(FP), R5
#else
	MOVD	R2, R4
	MOVD	R3, R5
#endif
	CMPBGE	R5, $32, clearge32

start:
	CMPBLE	R5, $3, clear0to3
	CMPBLE	R5, $7, clear4to7
	CMPBLE	R5, $11, clear8to11
	CMPBLE	R5, $15, clear12to15
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

clearge32:
	CMP	R5, $4096
	BLT	clear256Bto4KB

// For size >= 4KB, XC is loop unrolled 16 times (4KB = 256B * 16)
clearge4KB:
	XC	$256, 0(R4), 0(R4)
	XC	$256, 256(R4), 256(R4)
	XC	$256, 512(R4), 512(R4)
	XC	$256, 768(R4), 768(R4)
	XC	$256, 1024(R4), 1024(R4)
	XC	$256, 1280(R4), 1280(R4)
	XC	$256, 1536(R4), 1536(R4)
	XC	$256, 1792(R4), 1792(R4)
	XC	$256, 2048(R4), 2048(R4)
	XC	$256, 2304(R4), 2304(R4)
	XC	$256, 2560(R4), 2560(R4)
	XC	$256, 2816(R4), 2816(R4)
	XC	$256, 3072(R4), 3072(R4)
	XC	$256, 3328(R4), 3328(R4)
	XC	$256, 3584(R4), 3584(R4)
	XC	$256, 3840(R4), 3840(R4)
	ADD	$4096, R4
	ADD	$-4096, R5
	CMP	R5, $4096
	BGE	clearge4KB

clear256Bto4KB:
	CMP	R5, $256
	BLT	clear32to255
	XC	$256, 0(R4), 0(R4)
	ADD	$256, R4
	ADD	$-256, R5
	BR	clear256Bto4KB

clear32to255:
	CMPBEQ	R5, $0, done
	CMPBLT	R5, $32, start
	CMPBEQ	R5, $32, clear32
	CMPBLE	R5, $64, clear33to64
	CMP	R5, $128
	BLE	clear65to128
	CMP	R5, $255
	BLE	clear129to255

clear32:
	VZERO	V1
	VST	V1, 0(R4)
	VST	V1, 16(R4)
	RET

clear33to64:
	VZERO	V1
	VST	V1, 0(R4)
	VST	V1, 16(R4)
	ADD	$-32, R5
	VST	V1, 0(R4)(R5)
	VST	V1, 16(R4)(R5)
	RET

clear65to128:
	VZERO	V1
	VST	V1, 0(R4)
	VST	V1, 16(R4)
	VST	V1, 32(R4)
	VST	V1, 48(R4)
	ADD	$-64, R5
	VST	V1, 0(R4)(R5)
	VST	V1, 16(R4)(R5)
	VST	V1, 32(R4)(R5)
	VST	V1, 48(R4)(R5)
	RET

clear129to255:
	VZERO	V1
	VST	V1, 0(R4)
	VST	V1, 16(R4)
	VST	V1, 32(R4)
	VST	V1, 48(R4)
	VST	V1, 64(R4)
	VST	V1, 80(R4)
	VST	V1, 96(R4)
	VST	V1, 112(R4)
	ADD	$-128, R5
	VST	V1, 0(R4)(R5)
	VST	V1, 16(R4)(R5)
	VST	V1, 32(R4)(R5)
	VST	V1, 48(R4)(R5)
	VST	V1, 64(R4)(R5)
	VST	V1, 80(R4)(R5)
	VST	V1, 96(R4)(R5)
	VST	V1, 112(R4)(R5)
	RET

done:
	RET


// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0
	MOVW	$65536(R4), R5			// 1e020014de03800385f81000
	MOVW	$4096(R4), R5 			// 3e000014de03800385f81000
	MOVV	$65536(R4), R5			// 1e020014de03800385f81000
	MOVV	$4096(R4), R5			// 3e000014de03800385f81000
	ADD	$74565, R4			// 5e020014de178d0384781000
	ADD	$4097, R4  			// 3e000014de07800384781000
	ADDV	$74565, R4			// 5e020014de178d0384f81000
	ADDV	$4097, R4 			// 3e000014de07800384f81000
	AND	$74565, R4			// 5e020014de178d0384f81400
	AND	$4097, R4 		 	// 3e000014de07800384f81400
	ADD	$74565, R4, R5			// 5e020014de178d0385781000
	ADD	$4097, R4, R5  			// 3e000014de07800385781000
	ADDV	$74565, R4, R5			// 5e020014de178d0385f81000
	ADDV	$4097, R4, R5 			// 3e000014de07800385f81000
	AND	$74565, R4, R5			// 5e020014de178d0385f81400
	AND	$4097, R4, R5			// 3e000014de07800385f81400

	MOVW	R4, result+65540(FP)		// 1e020014de8f1000c4338029
	MOVW	R4, result+4097(FP)   		// 3e000014de8f1000c4278029
	MOVWU	R4, result+65540(FP)		// 1e020014de8f1000c4338029
	MOVWU	R4, result+4097(FP)  		// 3e000014de8f1000c4278029
	MOVV	R4, result+65540(FP)		// 1e020014de8f1000c433c029
	MOVV	R4, result+4097(FP)   		// 3e000014de8f1000c427c029
	MOVB	R4, result+65540(FP)		// 1e020014de8f1000c4330029
	MOVB	R4, result+4097(FP)   		// 3e000014de8f1000c4270029
	MOVBU	R4, result+65540(FP)		// 1e020014de8f1000c4330029
	MOVBU	R4, result+4097(FP)		// 3e000014de8f1000c4270029
	MOVW	R4, 65536(R5)			// 1e020014de971000c4038029
	MOVW	R4, 4096(R5)  			// 3e000014de971000c4038029
	MOVWU	R4, 65536(R5)			// 1e020014de971000c4038029
	MOVWU	R4, 4096(R5)			// 3e000014de971000c4038029
	MOVV	R4, 65536(R5)			// 1e020014de971000c403c029
	MOVV	R4, 4096(R5)			// 3e000014de971000c403c029
	MOVB	R4, 65536(R5)			// 1e020014de971000c4030029
	MOVB	R4, 4096(R5)			// 3e000014de971000c4030029
	MOVBU	R4, 65536(R5)			// 1e020014de971000c4030029
	MOVBU	R4, 4096(R5)			// 3e000014de971000c4030029
	SC	R4, 65536(R5)			// 1e020014de971000c4030021
	SC	R4, 4096(R5)	   		// 3e000014de971000c4030021
	MOVW	y+65540(FP), R4			// 1e020014de8f1000c4338028
	MOVWU	y+65540(FP), R4			// 1e020014de8f1000c433802a
	MOVV	y+65540(FP), R4			// 1e020014de8f1000c433c028
	MOVB	y+65540(FP), R4			// 1e020014de8f1000c4330028
	MOVBU	y+65540(FP), R4			// 1e020014de8f1000c433002a
	MOVW	y+4097(FP), R4			// 3e000014de8f1000c4278028
	MOVWU	y+4097(FP), R4			// 3e000014de8f1000c427802a
	MOVV	y+4097(FP), R4			// 3e000014de8f1000c427c028
	MOVB	y+4097(FP), R4			// 3e000014de8f1000c4270028
	MOVBU	y+4097(FP), R4			// 3e000014de8f1000c427002a
	MOVW	65536(R5), R4			// 1e020014de971000c4038028
	MOVWU	65536(R5), R4			// 1e020014de971000c403802a
	MOVV	65536(R5), R4			// 1e020014de971000c403c028
	MOVB	65536(R5), R4			// 1e020014de971000c4030028
	MOVBU	65536(R5), R4			// 1e020014de971000c403002a
	MOVW	4096(R5), R4			// 3e000014de971000c4038028
	MOVWU	4096(R5), R4			// 3e000014de971000c403802a
	MOVV	4096(R5), R4			// 3e000014de971000c403c028
	MOVB	4096(R5), R4			// 3e000014de971000c4030028
	MOVBU	4096(R5), R4			// 3e000014de971000c403002a
	MOVF	y+65540(FP), F4			// 1e020014de8f1000c433002b
	MOVD	y+65540(FP), F4			// 1e020014de8f1000c433802b
	MOVF	y+4097(FP), F4			// 3e000014de8f1000c427002b
	MOVD	y+4097(FP), F4			// 3e000014de8f1000c427802b
	MOVF	65536(R5), F4			// 1e020014de971000c403002b
	MOVD	65536(R5), F4			// 1e020014de971000c403802b
	MOVF	4096(R5), F4			// 3e000014de971000c403002b
	MOVD	4096(R5), F4			// 3e000014de971000c403802b
	MOVF	F4, result+65540(FP)		// 1e020014de8f1000c433402b
	MOVD	F4, result+65540(FP)		// 1e020014de8f1000c433c02b
	MOVF	F4, result+4097(FP)		// 3e000014de8f1000c427402b
	MOVD	F4, result+4097(FP)		// 3e000014de8f1000c427c02b
	MOVF	F4, 65536(R5)			// 1e020014de971000c403402b
	MOVD	F4, 65536(R5)			// 1e020014de971000c403c02b
	MOVF	F4, 4096(R5)			// 3e000014de971000c403402b
	MOVD	F4, 4096(R5)			// 3e000014de971000c403c02b

	MOVH	R4, result+65540(FP)		// 1e020014de8f1000c4334029
	MOVH	R4, 65536(R5)			// 1e020014de971000c4034029
	MOVH	y+65540(FP), R4			// 1e020014de8f1000c4334028
	MOVH	65536(R5), R4			// 1e020014de971000c4034028
	MOVH	R4, result+4097(FP)		// 3e000014de8f1000c4274029
	MOVH	R4, 4096(R5)			// 3e000014de971000c4034029
	MOVH	y+4097(FP), R4			// 3e000014de8f1000c4274028
	MOVH	4096(R5), R4			// 3e000014de971000c4034028
	MOVHU	R4, result+65540(FP)		// 1e020014de8f1000c4334029
	MOVHU	R4, 65536(R5)			// 1e020014de971000c4034029
	MOVHU	y+65540(FP), R4			// 1e020014de8f1000c433402a
	MOVHU	65536(R5), R4			// 1e020014de971000c403402a
	MOVHU	R4, result+4097(FP)		// 3e000014de8f1000c4274029
	MOVHU	R4, 4096(R5)			// 3e000014de971000c4034029
	MOVHU	y+4097(FP), R4 			// 3e000014de8f1000c427402a
	MOVHU	4096(R5), R4			// 3e000014de971000c403402a
	SGT	$74565, R4 			// 5e020014de178d0384781200
	SGT	$74565, R4, R5 			// 5e020014de178d0385781200
	SGT	$4097, R4 			// 3e000014de07800384781200
	SGT	$4097, R4, R5 			// 3e000014de07800385781200
	SGTU	$74565, R4 			// 5e020014de178d0384f81200
	SGTU	$74565, R4, R5 			// 5e020014de178d0385f81200
	SGTU	$4097, R4 			// 3e000014de07800384f81200
	SGTU	$4097, R4, R5 			// 3e000014de07800385f81200
	ADDU	$74565, R4 			// 5e020014de178d0384781000
	ADDU	$74565, R4, R5 			// 5e020014de178d0385781000
	ADDU	$4097, R4 			// 3e000014de07800384781000
	ADDU	$4097, R4, R5 			// 3e000014de07800385781000
	ADDVU	$4097, R4			// 3e000014de07800384f81000
	ADDVU	$4097, R4, R5 			// 3e000014de07800385f81000
	ADDVU	$74565, R4			// 5e020014de178d0384f81000
	ADDVU	$74565, R4, R5			// 5e020014de178d0385f81000
	OR	$74565, R4			// 5e020014de178d0384781500
	OR	$74565, R4, R5			// 5e020014de178d0385781500
	OR	$4097, R4			// 3e000014de07800384781500
	OR	$4097, R4, R5			// 3e000014de07800385781500
	XOR	$74565, R4			// 5e020014de178d0384f81500
	XOR	$74565, R4, R5			// 5e020014de178d0385f81500
	XOR	$4097, R4			// 3e000014de07800384f81500
	XOR	$4097, R4, R5			// 3e000014de07800385f81500

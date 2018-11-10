// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../../../../../runtime/textflag.h"

TEXT	foo(SB), DUPOK|NOSPLIT, $0

	ADDF	F0, F1, F2    // 002a31ee
	ADDD.EQ	F3, F4, F5    // 035b340e
	ADDF.NE	F0, F2        // 002a321e
	ADDD	F3, F5        // 035b35ee
	SUBF	F0, F1, F2    // 402a31ee
	SUBD.EQ	F3, F4, F5    // 435b340e
	SUBF.NE	F0, F2        // 402a321e
	SUBD	F3, F5        // 435b35ee
	MULF	F0, F1, F2    // 002a21ee
	MULD.EQ	F3, F4, F5    // 035b240e
	MULF.NE	F0, F2        // 002a221e
	MULD	F3, F5        // 035b25ee
	DIVF	F0, F1, F2    // 002a81ee
	DIVD.EQ	F3, F4, F5    // 035b840e
	DIVF.NE	F0, F2        // 002a821e
	DIVD	F3, F5        // 035b85ee
	NEGF	F0, F1        // 401ab1ee
	NEGD	F4, F5        // 445bb1ee
	ABSF	F0, F1        // c01ab0ee
	ABSD	F4, F5        // c45bb0ee
	SQRTF	F0, F1        // c01ab1ee
	SQRTD	F4, F5        // c45bb1ee
	MOVFD	F0, F1        // c01ab7ee
	MOVDF	F4, F5        // c45bb7ee

	LDREX	(R8), R9      // 9f9f98e1
	LDREXD	(R11), R12    // 9fcfbbe1
	STREX	R3, (R4), R5  // STREX  (R4), R3, R5 // 935f84e1
	STREXD	R8, (R9), g   // STREXD (R9), R8, g  // 98afa9e1

	CMPF    F8, F9        // c89ab4ee10faf1ee
	CMPD.CS F4, F5        // c45bb42e10faf12e
	CMPF.VS F7            // c07ab56e10faf16e
	CMPD    F6            // c06bb5ee10faf1ee

	MOVW	R4, F8        // 104b08ee
	MOVW	F4, R8        // 108b14ee

	MOVF	(R4), F9                                  // 009a94ed
	MOVD.EQ	(R4), F9                                  // 009b940d
	MOVF.NE	(g), F3                                   // 003a9a1d
	MOVD	(g), F3                                   // 003b9aed
	MOVF	0x20(R3), F9       // MOVF 32(R3), F9     // 089a93ed
	MOVD.EQ	0x20(R4), F9       // MOVD.EQ 32(R4), F9  // 089b940d
	MOVF.NE	-0x20(g), F3       // MOVF.NE -32(g), F3  // 083a1a1d
	MOVD	-0x20(g), F3       // MOVD -32(g), F3     // 083b1aed
	MOVF	F9, (R4)                                  // 009a84ed
	MOVD.EQ	F9, (R4)                                  // 009b840d
	MOVF.NE	F3, (g)                                   // 003a8a1d
	MOVD	F3, (g)                                   // 003b8aed
	MOVF	F9, 0x20(R3)       // MOVF F9, 32(R3)     // 089a83ed
	MOVD.EQ	F9, 0x20(R4)       // MOVD.EQ F9, 32(R4)  // 089b840d
	MOVF.NE	F3, -0x20(g)       // MOVF.NE F3, -32(g)  // 083a0a1d
	MOVD	F3, -0x20(g)       // MOVD F3, -32(g)     // 083b0aed
	MOVF	0x00ffffff(R2), F1 // MOVF 16777215(R2), F1
	MOVD	0x00ffffff(R2), F1 // MOVD 16777215(R2), F1
	MOVF	F2, 0x00ffffff(R2) // MOVF F2, 16777215(R2)
	MOVD	F2, 0x00ffffff(R2) // MOVD F2, 16777215(R2)
	MOVF	F0, math·Exp(SB)   // MOVF F0, math.Exp(SB)
	MOVF	math·Exp(SB), F0   // MOVF math.Exp(SB), F0
	MOVD	F0, math·Exp(SB)   // MOVD F0, math.Exp(SB)
	MOVD	math·Exp(SB), F0   // MOVD math.Exp(SB), F0
	MOVF	F4, F5                                    // 445ab0ee
	MOVD	F6, F7                                    // 467bb0ee
	MOVFW	F6, F8                                    // c68abdee
	MOVFW	F6, R8                                    // c6fabdee108b1fee
	MOVFW.U	F6, F8                                    // c68abcee
	MOVFW.U	F6, R8                                    // c6fabcee108b1fee
	MOVDW	F6, F8                                    // c68bbdee
	MOVDW	F6, R8                                    // c6fbbdee108b1fee
	MOVDW.U	F6, F8                                    // c68bbcee
	MOVDW.U	F6, R8                                    // c6fbbcee108b1fee
	MOVWF	F6, F8                                    // c68ab8ee
	MOVWF	R6, F8                                    // 106b0feecf8ab8ee
	MOVWF.U	F6, F8                                    // 468ab8ee
	MOVWF.U	R6, F8                                    // 106b0fee4f8ab8ee
	MOVWD	F6, F8                                    // c68bb8ee
	MOVWD	R6, F8                                    // 106b0feecf8bb8ee
	MOVWD.U	F6, F8                                    // 468bb8ee
	MOVWD.U	R6, F8                                    // 106b0fee4f8bb8ee

	END

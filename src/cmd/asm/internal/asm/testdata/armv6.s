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

	END

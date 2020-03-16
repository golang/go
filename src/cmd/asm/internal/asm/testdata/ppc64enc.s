// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Initial set of opcode combinations based on
// improvements to processing of constant
// operands.

// Full set will be added at a later date.

#include "../../../../../runtime/textflag.h"

TEXT asmtest(SB),DUPOK|NOSPLIT,$0
	// move constants
	MOVD $1, R3                     // 38600001
	MOVD $-1, R4                    // 3880ffff
	MOVD $65535, R5                 // 6005ffff
	MOVD $65536, R6                 // 64060001
	MOVD $-32767, R5                // 38a08001
	MOVD $-32768, R6                // 38c08000
	MOVD $1234567, R5               // 6405001260a5d687

	// add constants
	ADD $1, R3                      // 38630001
	ADD $1, R3, R4                  // 38830001
	ADD $-1, R4                     // 3884ffff
	ADD $-1, R4, R5                 // 38a4ffff
	ADD $65535, R5                  // 601fffff7cbf2a14
	ADD $65535, R5, R6              // 601fffff7cdf2a14
	ADD $65536, R6                  // 3cc60001
	ADD $65536, R6, R7              // 3ce60001
	ADD $-32767, R5                 // 38a58001
	ADD $-32767, R5, R4             // 38858001
	ADD $-32768, R6                 // 38c68000
	ADD $-32768, R6, R5             // 38a68000
	ADD $1234567, R5                // 641f001263ffd6877cbf2a14
	ADD $1234567, R5, R6            // 641f001263ffd6877cdf2a14

	// and constants
	ANDCC $1, R3                    // 70630001
	ANDCC $1, R3, R4                // 70640001
	ANDCC $-1, R4                   // 3be0ffff7fe42039
	ANDCC $-1, R4, R5               // 3be0ffff7fe52039
	ANDCC $65535, R5                // 70a5ffff
	ANDCC $65535, R5, R6            // 70a6ffff
	ANDCC $65536, R6                // 74c60001
	ANDCC $65536, R6, R7            // 74c70001
	ANDCC $-32767, R5               // 3be080017fe52839
	ANDCC $-32767, R5, R4           // 3be080017fe42839
	ANDCC $-32768, R6               // 3be080007fe63039
	ANDCC $-32768, R5, R6           // 3be080007fe62839
	ANDCC $1234567, R5              // 641f001263ffd6877fe52839
	ANDCC $1234567, R5, R6          // 641f001263ffd6877fe62839

	// or constants
	OR $1, R3                       // 60630001
	OR $1, R3, R4                   // 60640001
	OR $-1, R4                      // 3be0ffff7fe42378
	OR $-1, R4, R5                  // 3be0ffff7fe52378
	OR $65535, R5                   // 60a5ffff
	OR $65535, R5, R6               // 60a6ffff
	OR $65536, R6                   // 64c60001
	OR $65536, R6, R7               // 64c70001
	OR $-32767, R5                  // 3be080017fe52b78
	OR $-32767, R5, R6              // 3be080017fe62b78
	OR $-32768, R6                  // 3be080007fe63378
	OR $-32768, R6, R7              // 3be080007fe73378
	OR $1234567, R5                 // 641f001263ffd6877fe52b78
	OR $1234567, R5, R3             // 641f001263ffd6877fe32b78

	// or constants
	XOR $1, R3                      // 68630001
	XOR $1, R3, R4                  // 68640001
	XOR $-1, R4                     // 3be0ffff7fe42278
	XOR $-1, R4, R5                 // 3be0ffff7fe52278
	XOR $65535, R5                  // 68a5ffff
	XOR $65535, R5, R6              // 68a6ffff
	XOR $65536, R6                  // 6cc60001
	XOR $65536, R6, R7              // 6cc70001
	XOR $-32767, R5                 // 3be080017fe52a78
	XOR $-32767, R5, R6             // 3be080017fe62a78
	XOR $-32768, R6                 // 3be080007fe63278
	XOR $-32768, R6, R7             // 3be080007fe73278
	XOR $1234567, R5                // 641f001263ffd6877fe52a78
	XOR $1234567, R5, R3            // 641f001263ffd6877fe32a78

	// load-and-reserve
	LBAR (R4)(R3*1),$1,R5           // 7ca32069
	LBAR (R4),$0,R5                 // 7ca02068
	LBAR (R3),R5                    // 7ca01868
	LHAR (R4)(R3*1),$1,R5           // 7ca320e9
	LHAR (R4),$0,R5                 // 7ca020e8
	LHAR (R3),R5                    // 7ca018e8
	LWAR (R4)(R3*1),$1,R5           // 7ca32029
	LWAR (R4),$0,R5                 // 7ca02028
	LWAR (R3),R5                    // 7ca01828
	LDAR (R4)(R3*1),$1,R5           // 7ca320a9
	LDAR (R4),$0,R5                 // 7ca020a8
	LDAR (R3),R5                    // 7ca018a8

	// float constants
	FMOVD $(0.0), F1                // f0210cd0
	FMOVD $(-0.0), F1               // f0210cd0fc200850
	RET

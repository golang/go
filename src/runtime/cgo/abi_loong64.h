// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Macros for transitioning from the host ABI to Go ABI0.
//
// These macros save and restore the callee-saved registers
// from the stack, but they don't adjust stack pointer, so
// the user should prepare stack space in advance.
// SAVE_R22_TO_R31(offset) saves R22 ~ R31 to the stack space
// of ((offset)+0*8)(R3) ~ ((offset)+9*8)(R3).
//
// SAVE_F24_TO_F31(offset) saves F24 ~ F31 to the stack space
// of ((offset)+0*8)(R3) ~ ((offset)+7*8)(R3).
//
// Note: g is R22

#define SAVE_R22_TO_R31(offset)	\
	MOVV	g,   ((offset)+(0*8))(R3)	\
	MOVV	R23, ((offset)+(1*8))(R3)	\
	MOVV	R24, ((offset)+(2*8))(R3)	\
	MOVV	R25, ((offset)+(3*8))(R3)	\
	MOVV	R26, ((offset)+(4*8))(R3)	\
	MOVV	R27, ((offset)+(5*8))(R3)	\
	MOVV	R28, ((offset)+(6*8))(R3)	\
	MOVV	R29, ((offset)+(7*8))(R3)	\
	MOVV	R30, ((offset)+(8*8))(R3)	\
	MOVV	R31, ((offset)+(9*8))(R3)

#define SAVE_F24_TO_F31(offset)	\
	MOVD	F24, ((offset)+(0*8))(R3)	\
	MOVD	F25, ((offset)+(1*8))(R3)	\
	MOVD	F26, ((offset)+(2*8))(R3)	\
	MOVD	F27, ((offset)+(3*8))(R3)	\
	MOVD	F28, ((offset)+(4*8))(R3)	\
	MOVD	F29, ((offset)+(5*8))(R3)	\
	MOVD	F30, ((offset)+(6*8))(R3)	\
	MOVD	F31, ((offset)+(7*8))(R3)

#define RESTORE_R22_TO_R31(offset)	\
	MOVV	((offset)+(0*8))(R3),  g	\
	MOVV	((offset)+(1*8))(R3), R23	\
	MOVV	((offset)+(2*8))(R3), R24	\
	MOVV	((offset)+(3*8))(R3), R25	\
	MOVV	((offset)+(4*8))(R3), R26	\
	MOVV	((offset)+(5*8))(R3), R27	\
	MOVV	((offset)+(6*8))(R3), R28	\
	MOVV	((offset)+(7*8))(R3), R29	\
	MOVV	((offset)+(8*8))(R3), R30	\
	MOVV	((offset)+(9*8))(R3), R31

#define RESTORE_F24_TO_F31(offset)	\
	MOVD	((offset)+(0*8))(R3), F24	\
	MOVD	((offset)+(1*8))(R3), F25	\
	MOVD	((offset)+(2*8))(R3), F26	\
	MOVD	((offset)+(3*8))(R3), F27	\
	MOVD	((offset)+(4*8))(R3), F28	\
	MOVD	((offset)+(5*8))(R3), F29	\
	MOVD	((offset)+(6*8))(R3), F30	\
	MOVD	((offset)+(7*8))(R3), F31

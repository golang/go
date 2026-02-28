// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Macros for transitioning from the host ABI to Go ABI0.
//
// These macros save and restore the callee-saved registers
// from the stack, but they don't adjust stack pointer, so
// the user should prepare stack space in advance.
// SAVE_R19_TO_R28(offset) saves R19 ~ R28 to the stack space
// of ((offset)+0*8)(RSP) ~ ((offset)+9*8)(RSP).
//
// SAVE_F8_TO_F15(offset) saves F8 ~ F15 to the stack space
// of ((offset)+0*8)(RSP) ~ ((offset)+7*8)(RSP).
//
// R29 is not saved because Go will save and restore it.

#define SAVE_R19_TO_R28(offset) \
	STP	(R19, R20), ((offset)+0*8)(RSP) \
	STP	(R21, R22), ((offset)+2*8)(RSP) \
	STP	(R23, R24), ((offset)+4*8)(RSP) \
	STP	(R25, R26), ((offset)+6*8)(RSP) \
	STP	(R27, g), ((offset)+8*8)(RSP)

#define RESTORE_R19_TO_R28(offset) \
	LDP	((offset)+0*8)(RSP), (R19, R20) \
	LDP	((offset)+2*8)(RSP), (R21, R22) \
	LDP	((offset)+4*8)(RSP), (R23, R24) \
	LDP	((offset)+6*8)(RSP), (R25, R26) \
	LDP	((offset)+8*8)(RSP), (R27, g) /* R28 */

#define SAVE_F8_TO_F15(offset) \
	FSTPD	(F8, F9), ((offset)+0*8)(RSP) \
	FSTPD	(F10, F11), ((offset)+2*8)(RSP) \
	FSTPD	(F12, F13), ((offset)+4*8)(RSP) \
	FSTPD	(F14, F15), ((offset)+6*8)(RSP)

#define RESTORE_F8_TO_F15(offset) \
	FLDPD	((offset)+0*8)(RSP), (F8, F9) \
	FLDPD	((offset)+2*8)(RSP), (F10, F11) \
	FLDPD	((offset)+4*8)(RSP), (F12, F13) \
	FLDPD	((offset)+6*8)(RSP), (F14, F15)


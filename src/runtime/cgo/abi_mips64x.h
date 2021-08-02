// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Macros for transitioning from the host ABI to Go ABI0.
//
// These save the frame pointer, so in general, functions that use
// these should have zero frame size to suppress the automatic frame
// pointer, though it's harmless to not do this.

#ifdef GOARCH_mips64le

// PUSH_REGS_HOST_TO_ABI0 prepares for transitioning from
// the host ABI to Go ABI0 code. It saves all registers that are
// callee-save in the host ABI and caller-save in Go ABI0 and prepares
// for entry to Go.
//
// Save R16-R23 R28 R29 R30 R31 F20-F31 registers for hardfloat.
// Save R16-R23 R28 R29 R30 R31 registers for softfloat.
#ifdef GOMIPS64_hardfloat
#define PUSH_REGS_FLOAT_PART()	\
	MOVF	F20, 120(R29)	\
	MOVF	F21, 128(R29)	\
	MOVF	F22, 136(R29)	\
	MOVF	F23, 144(R29)	\
	MOVF	F24, 152(R29)	\
	MOVF	F25, 160(R29)	\
	MOVF	F26, 168(R29)	\
	MOVF	F27, 176(R29)	\
	MOVF	F28, 184(R29)	\
	MOVF	F29, 192(R29)	\
	MOVF	F30, 200(R29)	\
	MOVF	F31, 208(R29)

#define POP_REGS_FLOAT_PART()	\
	MOVF	120(R29), F20	\
	MOVF	128(R29), F21	\
	MOVF	136(R29), F22	\
	MOVF	144(R29), F23	\
	MOVF	152(R29), F24	\
	MOVF	160(R29), F25	\
	MOVF	168(R29), F26	\
	MOVF	176(R29), F27	\
	MOVF	184(R29), F28	\
	MOVF	192(R29), F29	\
	MOVF	200(R29), F30	\
	MOVF	208(R29), F31
#endif

#define PUSH_REGS_HOST_TO_ABI0()	\
	MOVV	R16, 24(R29)	\
	MOVV	R17, 32(R29)	\
	MOVV	R18, 40(R29)	\
	MOVV	R19, 48(R29)	\
	MOVV	R20, 56(R29)	\
	MOVV	R21, 64(R29)	\
	MOVV	R22, 72(R29)	\
	MOVV	R23, 80(R29)	\
	MOVV	RSB, 88(R29)	\
	MOVV	R29, 96(R29)	\
	MOVV	g, 104(R29)	\
	MOVV	R31, 112(R29)	\
	PUSH_REGS_FLOAT_PART()

#define POP_REGS_HOST_TO_ABI0()	\
	MOVV	24(R29), R16	\
	MOVV	32(R29), R17	\
	MOVV	40(R29), R18	\
	MOVV	48(R29), R19	\
	MOVV	56(R29), R20	\
	MOVV	64(R29), R21	\
	MOVV	72(R29), R22	\
	MOVV	80(R29), R23	\
	MOVV	88(R29), RSB	\
	MOVV	96(R29), R29	\
	MOVV	104(R29), g	\
	MOVV	112(R29), R31	\
	POP_REGS_FLOAT_PART()
#endif

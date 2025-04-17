// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Macros for transitioning from the host ABI to Go ABI0.
//
// These save the frame pointer, so in general, functions that use
// these should have zero frame size to suppress the automatic frame
// pointer, though it's harmless to not do this.

#ifdef GOOS_windows

// REGS_HOST_TO_ABI0_STACK is the stack bytes used by
// PUSH_REGS_HOST_TO_ABI0.
#define REGS_HOST_TO_ABI0_STACK (28*8 + 8)

// PUSH_REGS_HOST_TO_ABI0 prepares for transitioning from
// the host ABI to Go ABI0 code. It saves all registers that are
// callee-save in the host ABI and caller-save in Go ABI0 and prepares
// for entry to Go.
//
// Save DI SI BP BX R12 R13 R14 R15 X6-X15 registers and the DF flag.
// Clear the DF flag for the Go ABI.
// MXCSR matches the Go ABI, so we don't have to set that,
// and Go doesn't modify it, so we don't have to save it.
#define PUSH_REGS_HOST_TO_ABI0()	\
	PUSHFQ			\
	CLD			\
	ADJSP	$(REGS_HOST_TO_ABI0_STACK - 8)	\
	MOVQ	DI, (0*0)(SP)	\
	MOVQ	SI, (1*8)(SP)	\
	MOVQ	BP, (2*8)(SP)	\
	MOVQ	BX, (3*8)(SP)	\
	MOVQ	R12, (4*8)(SP)	\
	MOVQ	R13, (5*8)(SP)	\
	MOVQ	R14, (6*8)(SP)	\
	MOVQ	R15, (7*8)(SP)	\
	MOVUPS	X6, (8*8)(SP)	\
	MOVUPS	X7, (10*8)(SP)	\
	MOVUPS	X8, (12*8)(SP)	\
	MOVUPS	X9, (14*8)(SP)	\
	MOVUPS	X10, (16*8)(SP)	\
	MOVUPS	X11, (18*8)(SP)	\
	MOVUPS	X12, (20*8)(SP)	\
	MOVUPS	X13, (22*8)(SP)	\
	MOVUPS	X14, (24*8)(SP)	\
	MOVUPS	X15, (26*8)(SP)

#define POP_REGS_HOST_TO_ABI0()	\
	MOVQ	(0*0)(SP), DI	\
	MOVQ	(1*8)(SP), SI	\
	MOVQ	(2*8)(SP), BP	\
	MOVQ	(3*8)(SP), BX	\
	MOVQ	(4*8)(SP), R12	\
	MOVQ	(5*8)(SP), R13	\
	MOVQ	(6*8)(SP), R14	\
	MOVQ	(7*8)(SP), R15	\
	MOVUPS	(8*8)(SP), X6	\
	MOVUPS	(10*8)(SP), X7	\
	MOVUPS	(12*8)(SP), X8	\
	MOVUPS	(14*8)(SP), X9	\
	MOVUPS	(16*8)(SP), X10	\
	MOVUPS	(18*8)(SP), X11	\
	MOVUPS	(20*8)(SP), X12	\
	MOVUPS	(22*8)(SP), X13	\
	MOVUPS	(24*8)(SP), X14	\
	MOVUPS	(26*8)(SP), X15	\
	ADJSP	$-(REGS_HOST_TO_ABI0_STACK - 8)	\
	POPFQ

#else
// SysV ABI

#define REGS_HOST_TO_ABI0_STACK (6*8)

// SysV MXCSR matches the Go ABI, so we don't have to set that,
// and Go doesn't modify it, so we don't have to save it.
// Both SysV and Go require DF to be cleared, so that's already clear.
// The SysV and Go frame pointer conventions are compatible.
#define PUSH_REGS_HOST_TO_ABI0()	\
	ADJSP	$(REGS_HOST_TO_ABI0_STACK)	\
	MOVQ	BP, (5*8)(SP)	\
	LEAQ	(5*8)(SP), BP	\
	MOVQ	BX, (0*8)(SP)	\
	MOVQ	R12, (1*8)(SP)	\
	MOVQ	R13, (2*8)(SP)	\
	MOVQ	R14, (3*8)(SP)	\
	MOVQ	R15, (4*8)(SP)

#define POP_REGS_HOST_TO_ABI0()	\
	MOVQ	(0*8)(SP), BX	\
	MOVQ	(1*8)(SP), R12	\
	MOVQ	(2*8)(SP), R13	\
	MOVQ	(3*8)(SP), R14	\
	MOVQ	(4*8)(SP), R15	\
	MOVQ	(5*8)(SP), BP	\
	ADJSP	$-(REGS_HOST_TO_ABI0_STACK)

#endif

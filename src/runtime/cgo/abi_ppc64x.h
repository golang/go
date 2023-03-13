// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Macros for transitioning from the host ABI to Go ABI
//
// On PPC64/ELFv2 targets, the following registers are callee
// saved when called from C. They must be preserved before
// calling into Go which does not preserve any of them.
//
//	R14-R31
//	CR2-4
//	VR20-31
//	F14-F31
//
// xcoff(aix) and ELFv1 are similar, but may only require a
// subset of these.
//
// These macros assume a 16 byte aligned stack pointer. This
// is required by ELFv1, ELFv2, and AIX PPC64.

#define SAVE_GPR_SIZE (18*8)
#define SAVE_GPR(offset)               \
	MOVD	R14, (offset+8*0)(R1)  \
	MOVD	R15, (offset+8*1)(R1)  \
	MOVD	R16, (offset+8*2)(R1)  \
	MOVD	R17, (offset+8*3)(R1)  \
	MOVD	R18, (offset+8*4)(R1)  \
	MOVD	R19, (offset+8*5)(R1)  \
	MOVD	R20, (offset+8*6)(R1)  \
	MOVD	R21, (offset+8*7)(R1)  \
	MOVD	R22, (offset+8*8)(R1)  \
	MOVD	R23, (offset+8*9)(R1)  \
	MOVD	R24, (offset+8*10)(R1) \
	MOVD	R25, (offset+8*11)(R1) \
	MOVD	R26, (offset+8*12)(R1) \
	MOVD	R27, (offset+8*13)(R1) \
	MOVD	R28, (offset+8*14)(R1) \
	MOVD	R29, (offset+8*15)(R1) \
	MOVD	g,   (offset+8*16)(R1) \
	MOVD	R31, (offset+8*17)(R1)

#define RESTORE_GPR(offset)            \
	MOVD	(offset+8*0)(R1), R14  \
	MOVD	(offset+8*1)(R1), R15  \
	MOVD	(offset+8*2)(R1), R16  \
	MOVD	(offset+8*3)(R1), R17  \
	MOVD	(offset+8*4)(R1), R18  \
	MOVD	(offset+8*5)(R1), R19  \
	MOVD	(offset+8*6)(R1), R20  \
	MOVD	(offset+8*7)(R1), R21  \
	MOVD	(offset+8*8)(R1), R22  \
	MOVD	(offset+8*9)(R1), R23  \
	MOVD	(offset+8*10)(R1), R24 \
	MOVD	(offset+8*11)(R1), R25 \
	MOVD	(offset+8*12)(R1), R26 \
	MOVD	(offset+8*13)(R1), R27 \
	MOVD	(offset+8*14)(R1), R28 \
	MOVD	(offset+8*15)(R1), R29 \
	MOVD	(offset+8*16)(R1), g   \
	MOVD	(offset+8*17)(R1), R31

#define SAVE_FPR_SIZE (18*8)
#define SAVE_FPR(offset)               \
	FMOVD	F14, (offset+8*0)(R1)  \
	FMOVD	F15, (offset+8*1)(R1)  \
	FMOVD	F16, (offset+8*2)(R1)  \
	FMOVD	F17, (offset+8*3)(R1)  \
	FMOVD	F18, (offset+8*4)(R1)  \
	FMOVD	F19, (offset+8*5)(R1)  \
	FMOVD	F20, (offset+8*6)(R1)  \
	FMOVD	F21, (offset+8*7)(R1)  \
	FMOVD	F22, (offset+8*8)(R1)  \
	FMOVD	F23, (offset+8*9)(R1)  \
	FMOVD	F24, (offset+8*10)(R1) \
	FMOVD	F25, (offset+8*11)(R1) \
	FMOVD	F26, (offset+8*12)(R1) \
	FMOVD	F27, (offset+8*13)(R1) \
	FMOVD	F28, (offset+8*14)(R1) \
	FMOVD	F29, (offset+8*15)(R1) \
	FMOVD	F30, (offset+8*16)(R1) \
	FMOVD	F31, (offset+8*17)(R1)

#define RESTORE_FPR(offset)            \
	FMOVD	(offset+8*0)(R1), F14  \
	FMOVD	(offset+8*1)(R1), F15  \
	FMOVD	(offset+8*2)(R1), F16  \
	FMOVD	(offset+8*3)(R1), F17  \
	FMOVD	(offset+8*4)(R1), F18  \
	FMOVD	(offset+8*5)(R1), F19  \
	FMOVD	(offset+8*6)(R1), F20  \
	FMOVD	(offset+8*7)(R1), F21  \
	FMOVD	(offset+8*8)(R1), F22  \
	FMOVD	(offset+8*9)(R1), F23  \
	FMOVD	(offset+8*10)(R1), F24 \
	FMOVD	(offset+8*11)(R1), F25 \
	FMOVD	(offset+8*12)(R1), F26 \
	FMOVD	(offset+8*13)(R1), F27 \
	FMOVD	(offset+8*14)(R1), F28 \
	FMOVD	(offset+8*15)(R1), F29 \
	FMOVD	(offset+8*16)(R1), F30 \
	FMOVD	(offset+8*17)(R1), F31

// Save and restore VR20-31 (aka VSR56-63). These
// macros must point to a 16B aligned offset.
#define SAVE_VR_SIZE (12*16)
#define SAVE_VR(offset, rtmp)         \
	MOVD	$(offset+16*0), rtmp  \
	STVX	V20, (rtmp)(R1)       \
	MOVD	$(offset+16*1), rtmp  \
	STVX	V21, (rtmp)(R1)       \
	MOVD	$(offset+16*2), rtmp  \
	STVX	V22, (rtmp)(R1)       \
	MOVD	$(offset+16*3), rtmp  \
	STVX	V23, (rtmp)(R1)       \
	MOVD	$(offset+16*4), rtmp  \
	STVX	V24, (rtmp)(R1)       \
	MOVD	$(offset+16*5), rtmp  \
	STVX	V25, (rtmp)(R1)       \
	MOVD	$(offset+16*6), rtmp  \
	STVX	V26, (rtmp)(R1)       \
	MOVD	$(offset+16*7), rtmp  \
	STVX	V27, (rtmp)(R1)       \
	MOVD	$(offset+16*8), rtmp  \
	STVX	V28, (rtmp)(R1)       \
	MOVD	$(offset+16*9), rtmp  \
	STVX	V29, (rtmp)(R1)       \
	MOVD	$(offset+16*10), rtmp \
	STVX	V30, (rtmp)(R1)       \
	MOVD	$(offset+16*11), rtmp \
	STVX	V31, (rtmp)(R1)

#define RESTORE_VR(offset, rtmp)      \
	MOVD	$(offset+16*0), rtmp  \
	LVX	(rtmp)(R1), V20       \
	MOVD	$(offset+16*1), rtmp  \
	LVX	(rtmp)(R1), V21       \
	MOVD	$(offset+16*2), rtmp  \
	LVX	(rtmp)(R1), V22       \
	MOVD	$(offset+16*3), rtmp  \
	LVX	(rtmp)(R1), V23       \
	MOVD	$(offset+16*4), rtmp  \
	LVX	(rtmp)(R1), V24       \
	MOVD	$(offset+16*5), rtmp  \
	LVX	(rtmp)(R1), V25       \
	MOVD	$(offset+16*6), rtmp  \
	LVX	(rtmp)(R1), V26       \
	MOVD	$(offset+16*7), rtmp  \
	LVX	(rtmp)(R1), V27       \
	MOVD	$(offset+16*8), rtmp  \
	LVX	(rtmp)(R1), V28       \
	MOVD	$(offset+16*9), rtmp  \
	LVX	(rtmp)(R1), V29       \
	MOVD	$(offset+16*10), rtmp \
	LVX	(rtmp)(R1), V30       \
	MOVD	$(offset+16*11), rtmp \
	LVX	(rtmp)(R1), V31

// LR and CR are saved in the caller's frame. The callee must
// make space for all other callee-save registers.
#define SAVE_ALL_REG_SIZE (SAVE_GPR_SIZE+SAVE_FPR_SIZE+SAVE_VR_SIZE)

// Stack a frame and save all callee-save registers following the
// host OS's ABI. Fortunately, this is identical for AIX, ELFv1, and
// ELFv2. All host ABIs require the stack pointer to maintain 16 byte
// alignment, and save the callee-save registers in the same places.
//
// To restate, R1 is assumed to be aligned when this macro is used.
// This assumes the caller's frame is compliant with the host ABI.
// CR and LR are saved into the caller's frame per the host ABI.
// R0 is initialized to $0 as expected by Go.
#define STACK_AND_SAVE_HOST_TO_GO_ABI(extra)                       \
	MOVD	LR, R0                                             \
	MOVD	R0, 16(R1)                                         \
	MOVW	CR, R0                                             \
	MOVD	R0, 8(R1)                                          \
	MOVDU	R1, -(extra)-FIXED_FRAME-SAVE_ALL_REG_SIZE(R1)     \
	SAVE_GPR(extra+FIXED_FRAME)                                \
	SAVE_FPR(extra+FIXED_FRAME+SAVE_GPR_SIZE)                  \
	SAVE_VR(extra+FIXED_FRAME+SAVE_GPR_SIZE+SAVE_FPR_SIZE, R0) \
	MOVD	$0, R0

// This unstacks the frame, restoring all callee-save registers
// as saved by STACK_AND_SAVE_HOST_TO_GO_ABI.
//
// R0 is not guaranteed to contain $0 after this macro.
#define UNSTACK_AND_RESTORE_GO_TO_HOST_ABI(extra)                     \
	RESTORE_GPR(extra+FIXED_FRAME)                                \
	RESTORE_FPR(extra+FIXED_FRAME+SAVE_GPR_SIZE)                  \
	RESTORE_VR(extra+FIXED_FRAME+SAVE_GPR_SIZE+SAVE_FPR_SIZE, R0) \
	ADD 	$(extra+FIXED_FRAME+SAVE_ALL_REG_SIZE), R1            \
	MOVD	16(R1), R0                                            \
	MOVD	R0, LR                                                \
	MOVD	8(R1), R0                                             \
	MOVW	R0, CR

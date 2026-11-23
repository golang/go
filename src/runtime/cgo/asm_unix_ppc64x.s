// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && unix && (ppc64 || ppc64le)

#include "go_asm.h"
#include "textflag.h"
#include "asm_ppc64x.h"
#include "abi_ppc64x.h"

#ifdef GO_PPC64X_HAS_FUNCDESC
// On AIX (XCOFF), C function pointers are 24-byte descriptors, and the frame
// provides the 48-byte AIX linkage area (TOC at 40(R1)) plus 64-byte parameter
// save area before local save slots at 112(R1). As in runtime's
// callCfunction<>, cgoCallFrame.fn is the address of a TOC entry holding the
// function descriptor's address, as the address of a dynamic import is.
DEFINE_PPC64X_FUNCDESC(·cgoCallTrampoline, _cgoCallTrampoline<>)
TEXT _cgoCallTrampoline<>(SB),NOSPLIT|NOFRAME,$0-0
	// Save LR in the AIX LR save slot of the caller's frame, then
	// allocate the frame and store the back-chain pointer at 0(R1).
	MOVD	LR, R0
	MOVD	R0, 16(R1)
	MOVDU	R1, -128(R1)
	XOR	R0, R0
	MOVD	R2, 40(R1)
	MOVD	R14, 112(R1)
	MOVD	R3, R14
	MOVD	cgoCallFrame_fn(R14), R12
	MOVD	0(R12), R12
	MOVD	0(R12), R0
	MOVD	8(R12), R2
	MOVD	16(R12), R11
	MOVD	R0, CTR
	MOVD	cgoCallFrame_a0(R14), R3
	MOVD	cgoCallFrame_a1(R14), R4
	MOVD	cgoCallFrame_a2(R14), R5
	MOVD	cgoCallFrame_a3(R14), R6
	MOVD	cgoCallFrame_a4(R14), R7
	MOVD	cgoCallFrame_a5(R14), R8
	BL	(CTR)
	XOR	R0, R0
	MOVD	40(R1), R2
	MOVD	R3, cgoCallFrame_r1(R14)
	MOVD	$0, R3
	MOVD	112(R1), R14
	MOVD	144(R1), R0
	MOVD	R0, LR
	XOR	R0, R0
	ADD	$128, R1
	RET
#else
// cgoCallTrampoline is called by runtime.asmcgocall or runtime.asmcgocall_no_g
// using the PowerPC64 ELFv2 C ABI with R3 pointing to a cgoCallFrame.
// The frame includes the 32-byte linkage area and 64-byte parameter save area
// required by variadic C functions.
TEXT ·cgoCallTrampoline(SB),NOSPLIT|NOFRAME,$0-0
	// Save LR in the ELFv2 LR save slot of the caller's frame, then
	// allocate the frame and store the back-chain pointer at 0(R1).
	MOVD	LR, R0
	MOVD	R0, 16(R1)
	MOVDU	R1, -112(R1)
	XOR	R0, R0
	MOVD	R2, 24(R1)
	MOVD	R14, 96(R1)
	MOVD	R3, R14
	MOVD	cgoCallFrame_fn(R14), R12
	MOVD	R12, CTR
	MOVD	cgoCallFrame_a0(R14), R3
	MOVD	cgoCallFrame_a1(R14), R4
	MOVD	cgoCallFrame_a2(R14), R5
	MOVD	cgoCallFrame_a3(R14), R6
	MOVD	cgoCallFrame_a4(R14), R7
	MOVD	cgoCallFrame_a5(R14), R8
	BL	(CTR)
	XOR	R0, R0
	MOVD	24(R1), R2
	MOVD	R3, cgoCallFrame_r1(R14)
	MOVD	$0, R3
	MOVD	96(R1), R14
	MOVD	128(R1), R0
	MOVD	R0, LR
	XOR	R0, R0
	ADD	$112, R1
	RET
#endif

#define CGO_PROLOGUE \
	STACK_AND_SAVE_HOST_TO_GO_ABI(32); \
	MOVD	R2, 24(R1); \
	MOVD	$0, g

#define CGO_EPILOGUE \
	MOVD	24(R1), R2; \
	UNSTACK_AND_RESTORE_GO_TO_HOST_ABI(32)

// Indirectly call Go targets via CTR to avoid the linker's static NOSPLIT
// stack overflow check: these entrypoints are only invoked from C on a
// full-size C/pthread stack after saving 544 bytes of host registers.
#define CALL_GO(fn) \
	MOVD	$fn(SB), R12; \
	MOVD	R12, CTR; \
	BL	(CTR)

TEXT _cgo_wait_runtime_init_done(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$·runtime_init_done(SB), R4
	MOVWZ	(R4), R4
	CMPW	R4, $2
	BNE	slow
	// Make the load of runtime_init_done an acquire load.
	LWSYNC
	MOVD	$·cgo_context_function(SB), R3
	MOVD	(R3), R3
	CMP	R3, $0
	BNE	slow
	RET
slow:
	CGO_PROLOGUE
	CALL_GO(·cgoWaitRuntimeInitDone)
	MOVD	FIXED_FRAME+0(R1), R3
	CGO_EPILOGUE
	RET

TEXT _cgo_release_context(SB),NOSPLIT|NOFRAME,$0-0
	CMP	R3, $0
	BEQ	done
	CGO_PROLOGUE
	MOVD	R3, FIXED_FRAME+0(R1)
	CALL_GO(·cgoReleaseContext)
	CGO_EPILOGUE
done:
	RET

#ifdef GO_PPC64X_HAS_FUNCDESC
DEFINE_PPC64X_FUNCDESC(·pthreadKeyDestructor, _pthreadKeyDestructor<>)
TEXT _pthreadKeyDestructor<>(SB),NOSPLIT|NOFRAME,$0-0
#else
TEXT ·pthreadKeyDestructor(SB),NOSPLIT|NOFRAME,$0-0
#endif
	MOVD	R3, R4
	MOVD	$0, R3
	MOVD	$0, R6
	JMP	crosscall2(SB)

#ifdef GO_PPC64X_HAS_FUNCDESC
DEFINE_PPC64X_FUNCDESC(·x_cgo_threadentry, _x_cgo_threadentry<>)
TEXT _x_cgo_threadentry<>(SB),NOSPLIT|NOFRAME,$0-0
#else
TEXT ·x_cgo_threadentry(SB),NOSPLIT|NOFRAME,$0-0
#endif
	CGO_PROLOGUE
	MOVD	R3, FIXED_FRAME+0(R1)
	CALL_GO(·threadentry)
	CGO_EPILOGUE
	MOVD	$0, R3
	RET

#ifdef GO_PPC64X_HAS_FUNCDESC
DEFINE_PPC64X_FUNCDESC(x_cgo_init_desc<>, x_cgo_init<>)
#endif
TEXT x_cgo_init<>(SB),NOSPLIT|NOFRAME,$0-0
	CGO_PROLOGUE
	MOVD	R3, FIXED_FRAME+0(R1)
	MOVD	R4, FIXED_FRAME+8(R1)
	// runtime.rt0_go does not pass the TLS arguments used by
	// cgoInitTLS on Android; R5 and R6 hold unrelated values.
	MOVD	$0, R5
	MOVD	R5, FIXED_FRAME+16(R1)
	MOVD	R5, FIXED_FRAME+24(R1)
	CALL_GO(·cgoInit)
	CGO_EPILOGUE
	RET
GLOBL _cgo_init(SB), NOPTR, $8
#ifdef GO_PPC64X_HAS_FUNCDESC
DATA _cgo_init+0(SB)/8, $x_cgo_init_desc<>(SB)
#else
DATA _cgo_init+0(SB)/8, $x_cgo_init<>(SB)

// x_cgo_callers is jumped to from runtime.cgoSigtramp on the signal stack.
TEXT ·x_cgo_callers(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	LR, R0
	MOVD	R0, 16(R1)
	MOVDU	R1, -176(R1)
	MOVD	R2, 24(R1)
	XOR	R0, R0

	MOVD	R0, 96(R1)	// arg.context = 0
	MOVD	R5, 104(R1)	// arg.sigContext = context
	MOVD	R7, 112(R1)	// arg.buf = cgoCallers
	MOVD	$32, R9
	MOVD	R9, 120(R1)	// arg.max = 32

	MOVD	R3, 128(R1)	// save sig
	MOVD	R4, 136(R1)	// save info
	MOVD	R5, 144(R1)	// save context
	MOVD	R8, 152(R1)	// save sigtramp
	MOVD	R6, 160(R1)	// save cgoTraceback

	MOVD	$·tsanAcquireFnPtr(SB), R12
	MOVD	(R12), R12
	MOVD	(R12), R12
	CMP	R12, $0
	BEQ	callers_acq_done
	LWSYNC
	MOVD	$·cgoSyncPtr(SB), R3
	MOVD	(R3), R3
	MOVD	R12, CTR
	BL	(CTR)
	XOR	R0, R0
	MOVD	24(R1), R2
callers_acq_done:
	ADD	$96, R1, R3	// &arg
	MOVD	160(R1), R12	// cgoTraceback
	MOVD	R12, CTR
	BL	(CTR)		// (*cgoTraceback)(&arg)
	XOR	R0, R0
	MOVD	24(R1), R2

	MOVD	$·tsanReleaseFnPtr(SB), R12
	MOVD	(R12), R12
	MOVD	(R12), R12
	CMP	R12, $0
	BEQ	callers_rel_done
	LWSYNC
	MOVD	$·cgoSyncPtr(SB), R3
	MOVD	(R3), R3
	MOVD	R12, CTR
	BL	(CTR)
	XOR	R0, R0
	MOVD	24(R1), R2
callers_rel_done:
	MOVD	128(R1), R3	// restore sig
	MOVD	136(R1), R4	// restore info
	MOVD	144(R1), R5	// restore context
	MOVD	152(R1), R12	// restore sigtramp

	ADD	$176, R1
	MOVD	16(R1), R9
	MOVD	R9, LR
	MOVD	R12, CTR
	JMP	(CTR)		// sigtramp(sig, info, context)

GLOBL _cgo_callers(SB), NOPTR, $8
DATA _cgo_callers+0(SB)/8, $·x_cgo_callers(SB)
#endif

#ifdef GOOS_aix
TEXT _cgo_topofstack(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$__cgo_topofstack(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

TEXT _GLOBAL__I_65535_0_runtime_cgo(SB),NOSPLIT|NOFRAME,$0-0
	MOVD	$_rt0_ppc64_aix_lib(SB), R12
	MOVD	R12, CTR
	BR	(CTR)
#endif

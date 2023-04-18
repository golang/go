// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"
#include "asm_ppc64x.h"

// _rt0_ppc64_aix is a function descriptor of the entrypoint function
// __start. This name is needed by cmd/link.
DEFINE_PPC64X_FUNCDESC(_rt0_ppc64_aix, __start<>)

// The starting function must return in the loader to
// initialise some libraries, especially libthread which
// creates the main thread and adds the TLS in R13
// R19 contains a function descriptor to the loader function
// which needs to be called.
// This code is similar to the __start function in C
TEXT __start<>(SB),NOSPLIT,$-8
	XOR R0, R0
	MOVD $libc___n_pthreads(SB), R4
	MOVD 0(R4), R4
	MOVD $libc___mod_init(SB), R5
	MOVD 0(R5), R5
	MOVD 0(R19), R0
	MOVD R2, 40(R1)
	MOVD 8(R19), R2
	MOVD R18, R3
	MOVD R0, CTR
	BL (CTR) // Return to AIX loader

	// Launch rt0_go
	MOVD 40(R1), R2
	MOVD R14, R3 // argc
	MOVD R15, R4 // argv
	BL _main(SB)


DEFINE_PPC64X_FUNCDESC(main, _main)
TEXT _main(SB),NOSPLIT,$-8
	MOVD $runtime·rt0_go(SB), R12
	MOVD R12, CTR
	BR (CTR)


TEXT _rt0_ppc64_aix_lib(SB),NOSPLIT,$-8
	// Start with standard C stack frame layout and linkage.
	MOVD	LR, R0
	MOVD	R0, 16(R1) // Save LR in caller's frame.
	MOVW	CR, R0	   // Save CR in caller's frame
	MOVD	R0, 8(R1)

	MOVDU	R1, -344(R1) // Allocate frame.

	// Preserve callee-save registers.
	MOVD	R14, 48(R1)
	MOVD	R15, 56(R1)
	MOVD	R16, 64(R1)
	MOVD	R17, 72(R1)
	MOVD	R18, 80(R1)
	MOVD	R19, 88(R1)
	MOVD	R20, 96(R1)
	MOVD	R21,104(R1)
	MOVD	R22, 112(R1)
	MOVD	R23, 120(R1)
	MOVD	R24, 128(R1)
	MOVD	R25, 136(R1)
	MOVD	R26, 144(R1)
	MOVD	R27, 152(R1)
	MOVD	R28, 160(R1)
	MOVD	R29, 168(R1)
	MOVD	g, 176(R1) // R30
	MOVD	R31, 184(R1)
	FMOVD	F14, 192(R1)
	FMOVD	F15, 200(R1)
	FMOVD	F16, 208(R1)
	FMOVD	F17, 216(R1)
	FMOVD	F18, 224(R1)
	FMOVD	F19, 232(R1)
	FMOVD	F20, 240(R1)
	FMOVD	F21, 248(R1)
	FMOVD	F22, 256(R1)
	FMOVD	F23, 264(R1)
	FMOVD	F24, 272(R1)
	FMOVD	F25, 280(R1)
	FMOVD	F26, 288(R1)
	FMOVD	F27, 296(R1)
	FMOVD	F28, 304(R1)
	FMOVD	F29, 312(R1)
	FMOVD	F30, 320(R1)
	FMOVD	F31, 328(R1)

	// Synchronous initialization.
	MOVD	$runtime·reginit(SB), R12
	MOVD	R12, CTR
	BL	(CTR)

	MOVBZ	runtime·isarchive(SB), R3	// Check buildmode = c-archive
	CMP		$0, R3
	BEQ		done

	MOVD	R14, _rt0_ppc64_aix_lib_argc<>(SB)
	MOVD	R15, _rt0_ppc64_aix_lib_argv<>(SB)

	MOVD	$runtime·libpreinit(SB), R12
	MOVD	R12, CTR
	BL	(CTR)

	// Create a new thread to do the runtime initialization and return.
	MOVD	_cgo_sys_thread_create(SB), R12
	CMP	$0, R12
	BEQ	nocgo
	MOVD	$_rt0_ppc64_aix_lib_go(SB), R3
	MOVD	$0, R4
	MOVD	R2, 40(R1)
	MOVD	8(R12), R2
	MOVD	(R12), R12
	MOVD	R12, CTR
	BL	(CTR)
	MOVD	40(R1), R2
	BR	done

nocgo:
	MOVD	$0x800000, R12					   // stacksize = 8192KB
	MOVD	R12, 8(R1)
	MOVD	$_rt0_ppc64_aix_lib_go(SB), R12
	MOVD	R12, 16(R1)
	MOVD	$runtime·newosproc0(SB),R12
	MOVD	R12, CTR
	BL	(CTR)

done:
	// Restore saved registers.
	MOVD	48(R1), R14
	MOVD	56(R1), R15
	MOVD	64(R1), R16
	MOVD	72(R1), R17
	MOVD	80(R1), R18
	MOVD	88(R1), R19
	MOVD	96(R1), R20
	MOVD	104(R1), R21
	MOVD	112(R1), R22
	MOVD	120(R1), R23
	MOVD	128(R1), R24
	MOVD	136(R1), R25
	MOVD	144(R1), R26
	MOVD	152(R1), R27
	MOVD	160(R1), R28
	MOVD	168(R1), R29
	MOVD	176(R1), g // R30
	MOVD	184(R1), R31
	FMOVD	196(R1), F14
	FMOVD	200(R1), F15
	FMOVD	208(R1), F16
	FMOVD	216(R1), F17
	FMOVD	224(R1), F18
	FMOVD	232(R1), F19
	FMOVD	240(R1), F20
	FMOVD	248(R1), F21
	FMOVD	256(R1), F22
	FMOVD	264(R1), F23
	FMOVD	272(R1), F24
	FMOVD	280(R1), F25
	FMOVD	288(R1), F26
	FMOVD	296(R1), F27
	FMOVD	304(R1), F28
	FMOVD	312(R1), F29
	FMOVD	320(R1), F30
	FMOVD	328(R1), F31

	ADD	$344, R1

	MOVD	8(R1), R0
	MOVFL	R0, $0xff
	MOVD	16(R1), R0
	MOVD	R0, LR
	RET

DEFINE_PPC64X_FUNCDESC(_rt0_ppc64_aix_lib_go, __rt0_ppc64_aix_lib_go)

TEXT __rt0_ppc64_aix_lib_go(SB),NOSPLIT,$0
	MOVD	_rt0_ppc64_aix_lib_argc<>(SB), R3
	MOVD	_rt0_ppc64_aix_lib_argv<>(SB), R4
	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

DATA _rt0_ppc64_aix_lib_argc<>(SB)/8, $0
GLOBL _rt0_ppc64_aix_lib_argc<>(SB),NOPTR, $8
DATA _rt0_ppc64_aix_lib_argv<>(SB)/8, $0
GLOBL _rt0_ppc64_aix_lib_argv<>(SB),NOPTR, $8

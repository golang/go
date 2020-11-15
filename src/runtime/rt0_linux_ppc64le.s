// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

TEXT _rt0_ppc64le_linux(SB),NOSPLIT,$0
	XOR R0, R0	  // Make sure R0 is zero before _main
	BR _main<>(SB)

TEXT _rt0_ppc64le_linux_lib(SB),NOSPLIT,$-8
	// Start with standard C stack frame layout and linkage.
	MOVD	LR, R0
	MOVD	R0, 16(R1) // Save LR in caller's frame.
	MOVW	CR, R0     // Save CR in caller's frame
	MOVD	R0, 8(R1)
	MOVDU	R1, -320(R1) // Allocate frame.

	// Preserve callee-save registers.
	MOVD	R14, 24(R1)
	MOVD	R15, 32(R1)
	MOVD	R16, 40(R1)
	MOVD	R17, 48(R1)
	MOVD	R18, 56(R1)
	MOVD	R19, 64(R1)
	MOVD	R20, 72(R1)
	MOVD	R21, 80(R1)
	MOVD	R22, 88(R1)
	MOVD	R23, 96(R1)
	MOVD	R24, 104(R1)
	MOVD	R25, 112(R1)
	MOVD	R26, 120(R1)
	MOVD	R27, 128(R1)
	MOVD	R28, 136(R1)
	MOVD	R29, 144(R1)
	MOVD	g, 152(R1) // R30
	MOVD	R31, 160(R1)
	FMOVD	F14, 168(R1)
	FMOVD	F15, 176(R1)
	FMOVD	F16, 184(R1)
	FMOVD	F17, 192(R1)
	FMOVD	F18, 200(R1)
	FMOVD	F19, 208(R1)
	FMOVD	F20, 216(R1)
	FMOVD	F21, 224(R1)
	FMOVD	F22, 232(R1)
	FMOVD	F23, 240(R1)
	FMOVD	F24, 248(R1)
	FMOVD	F25, 256(R1)
	FMOVD	F26, 264(R1)
	FMOVD	F27, 272(R1)
	FMOVD	F28, 280(R1)
	FMOVD	F29, 288(R1)
	FMOVD	F30, 296(R1)
	FMOVD	F31, 304(R1)

	MOVD	R3, _rt0_ppc64le_linux_lib_argc<>(SB)
	MOVD	R4, _rt0_ppc64le_linux_lib_argv<>(SB)

	// Synchronous initialization.
	MOVD	$runtime·reginit(SB), R12
	MOVD	R12, CTR
	BL	(CTR)
	MOVD	$runtime·libpreinit(SB), R12
	MOVD	R12, CTR
	BL	(CTR)

	// Create a new thread to do the runtime initialization and return.
	MOVD	_cgo_sys_thread_create(SB), R12
	CMP	$0, R12
	BEQ	nocgo
	MOVD	$_rt0_ppc64le_linux_lib_go(SB), R3
	MOVD	$0, R4
	MOVD	R12, CTR
	BL	(CTR)
	BR	done

nocgo:
	MOVD	$0x800000, R12                     // stacksize = 8192KB
	MOVD	R12, 8(R1)
	MOVD	$_rt0_ppc64le_linux_lib_go(SB), R12
	MOVD	R12, 16(R1)
	MOVD	$runtime·newosproc0(SB),R12
	MOVD	R12, CTR
	BL	(CTR)

done:
	// Restore saved registers.
	MOVD	24(R1), R14
	MOVD	32(R1), R15
	MOVD	40(R1), R16
	MOVD	48(R1), R17
	MOVD	56(R1), R18
	MOVD	64(R1), R19
	MOVD	72(R1), R20
	MOVD	80(R1), R21
	MOVD	88(R1), R22
	MOVD	96(R1), R23
	MOVD	104(R1), R24
	MOVD	112(R1), R25
	MOVD	120(R1), R26
	MOVD	128(R1), R27
	MOVD	136(R1), R28
	MOVD	144(R1), R29
	MOVD	152(R1), g // R30
	MOVD	160(R1), R31
	FMOVD	168(R1), F14
	FMOVD	176(R1), F15
	FMOVD	184(R1), F16
	FMOVD	192(R1), F17
	FMOVD	200(R1), F18
	FMOVD	208(R1), F19
	FMOVD	216(R1), F20
	FMOVD	224(R1), F21
	FMOVD	232(R1), F22
	FMOVD	240(R1), F23
	FMOVD	248(R1), F24
	FMOVD	256(R1), F25
	FMOVD	264(R1), F26
	FMOVD	272(R1), F27
	FMOVD	280(R1), F28
	FMOVD	288(R1), F29
	FMOVD	296(R1), F30
	FMOVD	304(R1), F31

	ADD	$320, R1
	MOVD	8(R1), R0
	MOVFL	R0, $0xff
	MOVD	16(R1), R0
	MOVD	R0, LR
	RET

TEXT _rt0_ppc64le_linux_lib_go(SB),NOSPLIT,$0
	MOVD	_rt0_ppc64le_linux_lib_argc<>(SB), R3
	MOVD	_rt0_ppc64le_linux_lib_argv<>(SB), R4
	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

DATA _rt0_ppc64le_linux_lib_argc<>(SB)/8, $0
GLOBL _rt0_ppc64le_linux_lib_argc<>(SB),NOPTR, $8
DATA _rt0_ppc64le_linux_lib_argv<>(SB)/8, $0
GLOBL _rt0_ppc64le_linux_lib_argv<>(SB),NOPTR, $8

TEXT _main<>(SB),NOSPLIT,$-8
	// In a statically linked binary, the stack contains argc,
	// argv as argc string pointers followed by a NULL, envv as a
	// sequence of string pointers followed by a NULL, and auxv.
	// There is no TLS base pointer.
	//
	// In a dynamically linked binary, r3 contains argc, r4
	// contains argv, r5 contains envp, r6 contains auxv, and r13
	// contains the TLS pointer.
	//
	// Figure out which case this is by looking at r4: if it's 0,
	// we're statically linked; otherwise we're dynamically
	// linked.
	CMP	R0, R4
	BNE	dlink

	// Statically linked
	MOVD	0(R1), R3 // argc
	ADD	$8, R1, R4 // argv
	MOVD	$runtime·m0+m_tls(SB), R13 // TLS
	ADD	$0x7000, R13

dlink:
	BR	main(SB)

TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

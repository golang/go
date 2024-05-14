// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"
#include "asm_ppc64x.h"
#include "cgo/abi_ppc64x.h"

TEXT _rt0_ppc64le_linux(SB),NOSPLIT,$0
	XOR R0, R0	  // Make sure R0 is zero before _main
	BR _main<>(SB)

TEXT _rt0_ppc64le_linux_lib(SB),NOSPLIT|NOFRAME,$0
	// This is called with ELFv2 calling conventions. Convert to Go.
	// Allocate argument storage for call to newosproc0.
	STACK_AND_SAVE_HOST_TO_GO_ABI(16)

	MOVD	R3, _rt0_ppc64le_linux_lib_argc<>(SB)
	MOVD	R4, _rt0_ppc64le_linux_lib_argv<>(SB)

	// Synchronous initialization.
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
	MOVD	R12, 8+FIXED_FRAME(R1)
	MOVD	$_rt0_ppc64le_linux_lib_go(SB), R12
	MOVD	R12, 16+FIXED_FRAME(R1)
	MOVD	$runtime·newosproc0(SB),R12
	MOVD	R12, CTR
	BL	(CTR)

done:
	// Restore and return to ELFv2 caller.
	UNSTACK_AND_RESTORE_GO_TO_HOST_ABI(16)
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
	// The TLS pointer should be initialized to 0.
	//
	// In an ELFv2 compliant dynamically linked binary, R3 contains argc,
	// R4 contains argv, R5 contains envp, R6 contains auxv, and R13
	// contains the TLS pointer.
	//
	// When loading via glibc, the first doubleword on the stack points
	// to NULL a value. (that is *(uintptr)(R1) == 0). This is used to
	// differentiate static vs dynamically linked binaries.
	//
	// If loading with the musl loader, it doesn't follow the ELFv2 ABI. It
	// passes argc/argv similar to the linux kernel, R13 (TLS) is
	// initialized, and R3/R4 are undefined.
	MOVD	(R1), R12
	CMP	R12, $0
	BEQ	tls_and_argcv_in_reg

	// Arguments are passed via the stack (musl loader or a static binary)
	MOVD	0(R1), R3 // argc
	ADD	$8, R1, R4 // argv

	// Did the TLS pointer get set? If so, don't change it (e.g musl).
	CMP	R13, $0
	BNE	tls_and_argcv_in_reg

	MOVD	$runtime·m0+m_tls(SB), R13 // TLS
	ADD	$0x7000, R13

tls_and_argcv_in_reg:
	BR	main(SB)

TEXT main(SB),NOSPLIT,$-8
	MOVD	$runtime·rt0_go(SB), R12
	MOVD	R12, CTR
	BR	(CTR)

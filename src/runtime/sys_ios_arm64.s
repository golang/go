// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"
#include "cgo/abi_arm64.h"

TEXT runtime·chdir_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0	// arg 1 path
	BL	libc_chdir(SB)
	RET

TEXT runtime·cfBundleGetMainBundle_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19
	BL	libc_CFBundleGetMainBundle(SB)
	MOVD	R0, 0(R19)
	RET

TEXT runtime·cfBundleCopyBundleURL_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19
	MOVD	0(R0), R0	// arg 1 bundle
	BL	libc_CFBundleCopyBundleURL(SB)
	MOVD	R0, 8(R19)
	RET

TEXT runtime·cfURLGetFileSystemRepresentation_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19
	MOVD	8(R0), R1	// arg 2 resolveAgainstBase
	MOVD	16(R0), R2	// arg 3 path
	MOVD	24(R0), R3	// arg 4 pathLen
	MOVD	0(R0), R0	// arg 1 url
	BL	libc_CFURLGetFileSystemRepresentation(SB)
	MOVD	R0, 32(R19)
	RET

TEXT runtime·cfStringCreateWithCString_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19
	MOVD	8(R0), R1	// arg 2 str
	MOVD	16(R0), R2	// arg 3 encoding
	MOVD	0(R0), R0	// arg 1 alloc
	BL	libc_CFStringCreateWithCString(SB)
	MOVD	R0, 24(R19)
	RET

TEXT runtime·cfBundleGetValueForInfoDictionaryKey_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19
	MOVD	8(R0), R1	// arg 2 key
	MOVD	0(R0), R0	// arg 1 bundle
	BL	libc_CFBundleGetValueForInfoDictionaryKey(SB)
	MOVD	R0, 16(R19)
	RET

TEXT runtime·cfStringGetCString_trampoline(SB),NOSPLIT,$0
	MOVD	R0, R19
	MOVD	8(R0), R1	// arg 2 buf
	MOVD	16(R0), R2	// arg 3 bufLen
	MOVD	24(R0), R3	// arg 4 encoding
	MOVD	0(R0), R0	// arg 1 str
	BL	libc_CFStringGetCString(SB)
	MOVD	R0, 32(R19)
	RET

TEXT runtime·cfRelease_trampoline(SB),NOSPLIT,$0
	MOVD	0(R0), R0	// arg 1 ref
	BL	libc_CFRelease(SB)
	RET

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "go_tls.h"
#include "textflag.h"

TEXT _rt0_amd64_windows(SB),NOSPLIT,$0
	JMP	_rt0_amd64(SB)

// When building with -buildmode=(c-shared or c-archive), this
// symbol is called.
TEXT _rt0_amd64_windows_lib(SB),NOSPLIT,$0
	JMP	_rt0_amd64_lib(SB)

// _rt0_amd64_windows_plugin is the DllMain-equivalent entry point for
// -buildmode=plugin DLLs on windows/amd64. Plugin DLLs share the
// host's runtime, so the plugin's own runtime must NOT initialize.
// The OS calls this with the platform ABI: RCX=hinstDLL, RDX=fdwReason,
// R8=lpReserved. We just return 1 (TRUE) to signal success.
TEXT _rt0_amd64_windows_plugin(SB),NOSPLIT|NOFRAME,$0
	MOVQ	$1, AX
	RET

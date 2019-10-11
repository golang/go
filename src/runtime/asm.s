// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// funcdata for functions with no local variables in frame.
// Define two zero-length bitmaps, because the same index is used
// for the local variables as for the argument frame, and assembly
// frames have two argument bitmaps, one without results and one with results.
DATA runtime·no_pointers_stackmap+0x00(SB)/4, $2
DATA runtime·no_pointers_stackmap+0x04(SB)/4, $0
GLOBL runtime·no_pointers_stackmap(SB),RODATA, $8

// NaCl requires that these skips be verifiable machine code.
#ifdef GOARCH_amd64
#define SKIP4 BYTE $0x90; BYTE $0x90; BYTE $0x90; BYTE $0x90
#endif
#ifdef GOARCH_386
#define SKIP4 BYTE $0x90; BYTE $0x90; BYTE $0x90; BYTE $0x90
#endif
#ifdef GOARCH_wasm
#define SKIP4 UNDEF; UNDEF; UNDEF; UNDEF
#endif
#ifndef SKIP4
#define SKIP4 WORD $0
#endif

#define SKIP16 SKIP4; SKIP4; SKIP4; SKIP4
#define SKIP64 SKIP16; SKIP16; SKIP16; SKIP16

// This function must be sizeofSkipFunction bytes.
TEXT runtime·skipPleaseUseCallersFrames(SB),NOSPLIT,$0-0
	SKIP64; SKIP64; SKIP64; SKIP64

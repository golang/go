// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// FIXED_FRAME defines the size of the fixed part of a stack frame. A stack
// frame looks like this:
//
// +---------------------+
// | local variable area |
// +---------------------+
// | argument area       |
// +---------------------+ <- R1+FIXED_FRAME
// | fixed area          |
// +---------------------+ <- R1
//
// So a function that sets up a stack frame at all uses as least FIXED_FRAME
// bytes of stack. This mostly affects assembly that calls other functions
// with arguments (the arguments should be stored at FIXED_FRAME+0(R1),
// FIXED_FRAME+8(R1) etc) and some other low-level places.
//
// The reason for using a constant is to make supporting PIC easier (although
// we only support PIC on ppc64le which has a minimum 32 bytes of stack frame,
// and currently always use that much, PIC on ppc64 would need to use 48).

#define FIXED_FRAME 32

// aix/ppc64 uses XCOFF which uses function descriptors.
// AIX cannot perform the TOC relocation in a text section.
// Therefore, these descriptors must live in a data section.
#ifdef GOOS_aix
#ifdef GOARCH_ppc64
#define GO_PPC64X_HAS_FUNCDESC
#define DEFINE_PPC64X_FUNCDESC(funcname, localfuncname)	\
	DATA	funcname+0(SB)/8, $localfuncname(SB) 	\
	DATA	funcname+8(SB)/8, $TOC(SB)		\
	DATA	funcname+16(SB)/8, $0			\
	GLOBL	funcname(SB), NOPTR, $24
#endif
#endif

// linux/ppc64 uses ELFv1 which uses function descriptors.
// These must also look like ABI0 functions on linux/ppc64
// to work with abi.FuncPCABI0(sigtramp) in os_linux.go.
// Only static codegen is supported on linux/ppc64, so TOC
// is not needed.
#ifdef GOOS_linux
#ifdef GOARCH_ppc64
#define GO_PPC64X_HAS_FUNCDESC
#define DEFINE_PPC64X_FUNCDESC(funcname, localfuncname)	\
	TEXT	funcname(SB),NOSPLIT|NOFRAME,$0		\
		DWORD	$localfuncname(SB)		\
		DWORD	$0				\
		DWORD	$0
#endif
#endif

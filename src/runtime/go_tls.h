// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifdef GOOS_windows
// Offset of the TEB's ThreadLocalStoragePointer field, which points to the
// array of module TLS blocks indexed by _tls_index.
#ifdef GOARCH_386
#define TEB_ThreadLocalStoragePointer 0x2c
#else
#define TEB_ThreadLocalStoragePointer 0x58
#endif
#endif

#ifdef GOARCH_arm
#define LR R14
#endif

#ifdef GOARCH_amd64
#ifdef GOOS_windows
// get_tls loads the address of the current thread's g slot into r.
// It clobbers R11; r must differ from R11.
#define get_tls(r) \
	MOVL	_tls_index(SB), r /* 32-bit TLS index, zero-extended to 64 bits. */ \
	MOVQ	TEB_ThreadLocalStoragePointer(GS), R11 /* R11 is volatile in the host ABI. */ \
	MOVQ	(R11)(r*8), r \
	MOVQ	runtime·tls_g(SB), R11 \
	LEAQ	(r)(R11*1), r
#else
#define	get_tls(r)	MOVQ TLS, r
#endif
#define	g(r)	0(r)(TLS*1)
#endif

#ifdef GOARCH_386
#ifdef GOOS_windows
// get_tls2 lets callers preserve AX or load the TLS address into AX.
// r and scratch must differ; scratch is clobbered.
#define get_tls2(r, scratch) \
	MOVL	_tls_index(SB), r \
	MOVL	TEB_ThreadLocalStoragePointer(FS), scratch \
	MOVL	(scratch)(r*4), r \
	MOVL	runtime·tls_g(SB), scratch \
	LEAL	(r)(scratch*1), r
#define get_tls(r) get_tls2(r, AX)
#else
#define	get_tls(r)	MOVL TLS, r
#define	get_tls2(r, scratch)	MOVL TLS, r
#endif
#define	g(r)	0(r)(TLS*1)
#endif

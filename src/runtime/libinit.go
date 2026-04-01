// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64 || arm || arm64 || loong64 || ppc64 || ppc64le || riscv64 || s390x

package runtime

import (
	"internal/abi"
	"unsafe"
)

// rt0LibGoFn holds the function pointer to rt0_lib_go suitable for thread
// creation. On most platforms it is zero, meaning the raw code address should
// be used. On AIX it is set by libpreinit to a function descriptor pointer,
// because pthread_create on AIX expects a function descriptor, not a raw
// code address.
var rt0LibGoFn uintptr

// libInit is common startup code for most architectures when
// using -buildmode=c-archive or -buildmode=c-shared.
//
// May run with m.p==nil, so write barriers are not allowed.
//
//go:nowritebarrierrec
//go:nosplit
func libInit() {
	// Synchronous initialization.
	libpreinit()

	// Use the platform-specific function pointer if set (e.g. AIX
	// function descriptor), otherwise fall back to the raw code address.
	fn := unsafe.Pointer(rt0LibGoFn)
	if fn == nil {
		fn = unsafe.Pointer(abi.FuncPCABIInternal(rt0_lib_go))
	}

	// Asynchronous initialization.
	// Prefer creating a thread via cgo if it is available.
	if _cgo_sys_thread_create != nil {
		// No g because the TLS is not set up until later in rt0_go.
		asmcgocall_no_g(_cgo_sys_thread_create, fn)
	} else {
		const stackSize = 0x800000 // 8192KB
		newosproc0(stackSize, fn)
	}
}

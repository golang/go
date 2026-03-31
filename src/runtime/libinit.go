// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64 || arm || arm64 || loong64 || ppc64 || ppc64le || riscv64 || s390x

package runtime

import (
	"internal/abi"
	"unsafe"
)

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

	// Asynchronous initialization.
	// Prefer creating a thread via cgo if it is available.
	if _cgo_sys_thread_create != nil {
		// No g because the TLS is not set up until later in rt0_go.
		asmcgocall_no_g(_cgo_sys_thread_create, unsafe.Pointer(abi.FuncPCABIInternal(rt0_lib_go)))
	} else {
		const stackSize = 0x800000 // 8192KB
		newosproc0(stackSize, unsafe.Pointer(abi.FuncPCABIInternal(rt0_lib_go)))
	}
}

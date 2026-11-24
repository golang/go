// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && msan && ((linux && (amd64 || arm64 || loong64)) || (freebsd && amd64))

package cgo

import "unsafe"

//go:cgo_import_static __msan_write_go
//go:linkname _msan_write_go __msan_write_go
var _msan_write_go byte

// msanEnabled is emitted into go_asm.h as const_msanEnabled so assembly files
// compile out MSAN unpoisoning when -msan is not active.
const msanEnabled = 1

// msanUnpoisonFn is used by x_cgo_call_context and x_cgo_callers in assembly.
var msanUnpoisonFn = unsafe.Pointer(&_msan_write_go)

//go:linkname msanUnpoison runtime.msanwrite
//go:noescape
func msanUnpoison(addr unsafe.Pointer, size uintptr)

// msanUnpoisonMode unpoisons [addr, addr+size). When noG is true it calls
// __msan_write_go on the current stack without touching g or g0.
//
//go:noinline
//go:nosplit
//go:norace
//go:nocheckptr
func msanUnpoisonMode(noG bool, addr unsafe.Pointer, size uintptr) {
	if noG {
		var f cgoCallFrame
		f.fn, f.a0, f.a1 = uintptr(msanUnpoisonFn), uintptr(addr), size
		runtime_asmcgocall_no_g(cgoCallTrampolinePC(), unsafe.Pointer(&f))
		return
	}
	msanUnpoison(addr, size)
}

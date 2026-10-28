// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || (linux && !android && (amd64 || arm64))

package cgo

import (
	"internal/cpu"
	"unsafe"
)

const offsetX86HasAVX = unsafe.Offsetof(cpu.X86.HasAVX)

//go:linkname _cgo_tsan_acquire_fn _cgo_tsan_acquire_fn
var _cgo_tsan_acquire_fn uintptr

//go:linkname _cgo_tsan_release_fn _cgo_tsan_release_fn
var _cgo_tsan_release_fn uintptr

//go:linkname _cgo_msan_unpoison_fn _cgo_msan_unpoison_fn
var _cgo_msan_unpoison_fn uintptr

//go:nosplit
//go:norace
func tsanAcquire() {
	if _cgo_tsan_acquire_fn != 0 {
		callC(_cgo_tsan_acquire_fn, 0, 0, 0, 0, 0, 0)
	}
}

//go:nosplit
//go:norace
func tsanRelease() {
	if _cgo_tsan_release_fn != 0 {
		callC(_cgo_tsan_release_fn, 0, 0, 0, 0, 0, 0)
	}
}

//go:nosplit
//go:norace
func msanUnpoison(addr unsafe.Pointer, size uintptr) {
	if _cgo_msan_unpoison_fn != 0 {
		callC(_cgo_msan_unpoison_fn, uintptr(addr), size, 0, 0, 0, 0)
	}
}

// callC calls a C function pointer fn with up to 6 integer/pointer arguments
// using the platform C ABI and returns its integer/pointer result.
//
//go:noescape
func callC(fn, a0, a1, a2, a3, a4, a5 uintptr) uintptr

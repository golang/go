// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && darwin

package cgo

import "unsafe"

//go:linkname _libc_pthread_get_stackaddr_np libc_pthread_get_stackaddr_np
var _libc_pthread_get_stackaddr_np byte
var libc_pthread_get_stackaddr_np = unsafe.Pointer(&_libc_pthread_get_stackaddr_np)

//go:linkname _libc_pthread_get_stacksize_np libc_pthread_get_stacksize_np
var _libc_pthread_get_stacksize_np byte
var libc_pthread_get_stacksize_np = unsafe.Pointer(&_libc_pthread_get_stacksize_np)

// cgoGetStackBound returns the current OS thread's stack bounds [lo, hi).
// It must be called while pinned to the M (or on a system/C stack) so
// pthread_self refers to a fixed OS thread.
//
//go:nosplit
//go:norace
func cgoGetStackBound() (uintptr, uintptr) {
	self := asmcgocall(uintptr(libc_pthread_self), 0, 0, 0, 0, 0, 0)
	high := asmcgocall(uintptr(libc_pthread_get_stackaddr_np), self, 0, 0, 0, 0, 0)
	size := asmcgocall(uintptr(libc_pthread_get_stacksize_np), self, 0, 0, 0, 0, 0)
	return high - size, high
}

//go:cgo_import_dynamic libc_setenv setenv "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_unsetenv unsetenv "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_self pthread_self "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_get_stackaddr_np pthread_get_stackaddr_np "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_get_stacksize_np pthread_get_stacksize_np "/usr/lib/libSystem.B.dylib"

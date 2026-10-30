// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && darwin

package cgo

import "unsafe"

const (
	sigSetMask            = 3
	_EAGAIN               = 35
	pthreadCreateDetached = 2
)

//go:linkname _libc_pthread_attr_setstacksize libc_pthread_attr_setstacksize
var _libc_pthread_attr_setstacksize byte
var libc_pthread_attr_setstacksize = unsafe.Pointer(&_libc_pthread_attr_setstacksize)

//go:linkname _libc_pthread_get_stackaddr_np libc_pthread_get_stackaddr_np
var _libc_pthread_get_stackaddr_np byte
var libc_pthread_get_stackaddr_np = unsafe.Pointer(&_libc_pthread_get_stackaddr_np)

//go:linkname _libc_pthread_get_stacksize_np libc_pthread_get_stacksize_np
var _libc_pthread_get_stacksize_np byte
var libc_pthread_get_stacksize_np = unsafe.Pointer(&_libc_pthread_get_stacksize_np)

//go:nosplit
//go:norace
//go:nocheckptr
func cgoThreadStackSize(attr *pthreadAttr) uintptr {
	// Copy stack size from parent thread instead of using the
	// non-main thread default stack size.
	self := asmcgocall(uintptr(libc_pthread_self), 0, 0, 0, 0, 0, 0)
	size := asmcgocall(uintptr(libc_pthread_get_stacksize_np), self, 0, 0, 0, 0, 0)
	asmcgocall(uintptr(libc_pthread_attr_setstacksize), uintptr(unsafe.Pointer(attr)), size, 0, 0, 0, 0)
	return size
}

// cgoGetStackBound returns the current OS thread's stack bounds [lo, hi).
// It must be called while pinned to the M (or on a system/C stack) so
// pthread_self refers to a fixed OS thread.
//
//go:nosplit
//go:norace
func cgoGetStackBound() (uintptr, uintptr) {
	self := asmcgocallNoG(uintptr(libc_pthread_self), 0, 0, 0, 0, 0, 0)
	high := asmcgocallNoG(uintptr(libc_pthread_get_stackaddr_np), self, 0, 0, 0, 0, 0)
	size := asmcgocallNoG(uintptr(libc_pthread_get_stacksize_np), self, 0, 0, 0, 0, 0)
	return high - size, high
}

//go:cgo_import_dynamic libc_setenv setenv "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_unsetenv unsetenv "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_create pthread_create "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_init pthread_attr_init "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_setstacksize pthread_attr_setstacksize "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_attr_destroy pthread_attr_destroy "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_self pthread_self "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_get_stackaddr_np pthread_get_stackaddr_np "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_get_stacksize_np pthread_get_stacksize_np "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_pthread_sigmask pthread_sigmask "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic libc_sigfillset sigfillset "/usr/lib/libSystem.B.dylib"

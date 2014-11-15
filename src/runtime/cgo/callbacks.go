// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgo

import "unsafe"

// These utility functions are available to be called from code
// compiled with gcc via crosscall2.

// cgocallback is defined in runtime
//go:linkname _runtime_cgocallback runtime.cgocallback
func _runtime_cgocallback(unsafe.Pointer, unsafe.Pointer, uintptr)

// The declaration of crosscall2 is:
//   void crosscall2(void (*fn)(void *, int), void *, int);
//
// We need to export the symbol crosscall2 in order to support
// callbacks from shared libraries. This applies regardless of
// linking mode.
//go:cgo_export_static crosscall2
//go:cgo_export_dynamic crosscall2

// Allocate memory.  This allocates the requested number of bytes in
// memory controlled by the Go runtime.  The allocated memory will be
// zeroed.  You are responsible for ensuring that the Go garbage
// collector can see a pointer to the allocated memory for as long as
// it is valid, e.g., by storing a pointer in a local variable in your
// C function, or in memory allocated by the Go runtime.  If the only
// pointers are in a C global variable or in memory allocated via
// malloc, then the Go garbage collector may collect the memory.

// Call like this in code compiled with gcc:
//   struct { size_t len; void *ret; } a;
//   a.len = /* number of bytes to allocate */;
//   crosscall2(_cgo_allocate, &a, sizeof a);
//   /* Here a.ret is a pointer to the allocated memory.  */

//go:linkname _runtime_cgo_allocate_internal runtime._cgo_allocate_internal
var _runtime_cgo_allocate_internal byte

//go:linkname _cgo_allocate _cgo_allocate
//go:cgo_export_static _cgo_allocate
//go:cgo_export_dynamic _cgo_allocate
//go:nosplit
func _cgo_allocate(a unsafe.Pointer, n int32) {
	_runtime_cgocallback(unsafe.Pointer(&_runtime_cgo_allocate_internal), a, uintptr(n))
}

// Panic.  The argument is converted into a Go string.

// Call like this in code compiled with gcc:
//   struct { const char *p; } a;
//   a.p = /* string to pass to panic */;
//   crosscall2(_cgo_panic, &a, sizeof a);
//   /* The function call will not return.  */

//go:linkname _runtime_cgo_panic_internal runtime._cgo_panic_internal
var _runtime_cgo_panic_internal byte

//go:linkname _cgo_panic _cgo_panic
//go:cgo_export_static _cgo_panic
//go:cgo_export_dynamic _cgo_panic
//go:nosplit
func _cgo_panic(a unsafe.Pointer, n int32) {
	_runtime_cgocallback(unsafe.Pointer(&_runtime_cgo_panic_internal), a, uintptr(n))
}

//go:cgo_import_static x_cgo_init
//go:linkname x_cgo_init x_cgo_init
//go:linkname _cgo_init _cgo_init
var x_cgo_init byte
var _cgo_init = &x_cgo_init

//go:cgo_import_static x_cgo_malloc
//go:linkname x_cgo_malloc x_cgo_malloc
//go:linkname _cgo_malloc _cgo_malloc
var x_cgo_malloc byte
var _cgo_malloc = &x_cgo_malloc

//go:cgo_import_static x_cgo_free
//go:linkname x_cgo_free x_cgo_free
//go:linkname _cgo_free _cgo_free
var x_cgo_free byte
var _cgo_free = &x_cgo_free

//go:cgo_import_static x_cgo_thread_start
//go:linkname x_cgo_thread_start x_cgo_thread_start
//go:linkname _cgo_thread_start _cgo_thread_start
var x_cgo_thread_start byte
var _cgo_thread_start = &x_cgo_thread_start

//go:cgo_export_static _cgo_topofstack
//go:cgo_export_dynamic _cgo_topofstack

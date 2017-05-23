// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgo

import "unsafe"

// These utility functions are available to be called from code
// compiled with gcc via crosscall2.

// cgocallback is defined in runtime
//go:linkname _runtime_cgocallback runtime.cgocallback
func _runtime_cgocallback(unsafe.Pointer, unsafe.Pointer, uintptr, uintptr)

// The declaration of crosscall2 is:
//   void crosscall2(void (*fn)(void *, int), void *, int);
//
// We need to export the symbol crosscall2 in order to support
// callbacks from shared libraries. This applies regardless of
// linking mode.
//
// Compatibility note: crosscall2 actually takes four arguments, but
// it works to call it with three arguments when calling _cgo_panic.
// That is supported for backward compatibility.
//go:cgo_export_static crosscall2
//go:cgo_export_dynamic crosscall2

// Panic. The argument is converted into a Go string.

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
//go:norace
func _cgo_panic(a unsafe.Pointer, n int32) {
	_runtime_cgocallback(unsafe.Pointer(&_runtime_cgo_panic_internal), a, uintptr(n), 0)
}

//go:cgo_import_static x_cgo_init
//go:linkname x_cgo_init x_cgo_init
//go:linkname _cgo_init _cgo_init
var x_cgo_init byte
var _cgo_init = &x_cgo_init

//go:cgo_import_static x_cgo_thread_start
//go:linkname x_cgo_thread_start x_cgo_thread_start
//go:linkname _cgo_thread_start _cgo_thread_start
var x_cgo_thread_start byte
var _cgo_thread_start = &x_cgo_thread_start

// Creates a new system thread without updating any Go state.
//
// This method is invoked during shared library loading to create a new OS
// thread to perform the runtime initialization. This method is similar to
// _cgo_sys_thread_start except that it doesn't update any Go state.

//go:cgo_import_static x_cgo_sys_thread_create
//go:linkname x_cgo_sys_thread_create x_cgo_sys_thread_create
//go:linkname _cgo_sys_thread_create _cgo_sys_thread_create
var x_cgo_sys_thread_create byte
var _cgo_sys_thread_create = &x_cgo_sys_thread_create

// Notifies that the runtime has been initialized.
//
// We currently block at every CGO entry point (via _cgo_wait_runtime_init_done)
// to ensure that the runtime has been initialized before the CGO call is
// executed. This is necessary for shared libraries where we kickoff runtime
// initialization in a separate thread and return without waiting for this
// thread to complete the init.

//go:cgo_import_static x_cgo_notify_runtime_init_done
//go:linkname x_cgo_notify_runtime_init_done x_cgo_notify_runtime_init_done
//go:linkname _cgo_notify_runtime_init_done _cgo_notify_runtime_init_done
var x_cgo_notify_runtime_init_done byte
var _cgo_notify_runtime_init_done = &x_cgo_notify_runtime_init_done

// Sets the traceback context function. See runtime.SetCgoTraceback.

//go:cgo_import_static x_cgo_set_context_function
//go:linkname x_cgo_set_context_function x_cgo_set_context_function
//go:linkname _cgo_set_context_function _cgo_set_context_function
var x_cgo_set_context_function byte
var _cgo_set_context_function = &x_cgo_set_context_function

//go:cgo_export_static _cgo_topofstack
//go:cgo_export_dynamic _cgo_topofstack

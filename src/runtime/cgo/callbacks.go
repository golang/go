// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgo

import "unsafe"

// These utility functions are available to be called from code
// compiled with gcc via crosscall2.

// The declaration of crosscall2 is:
//   void crosscall2(void (*fn)(void *), void *, int);
//
// We need to export the symbol crosscall2 in order to support
// callbacks from shared libraries. This applies regardless of
// linking mode.
//
// Compatibility note: SWIG uses crosscall2 in exactly one situation:
// to call _cgo_panic using the pattern shown below. We need to keep
// that pattern working. In particular, crosscall2 actually takes four
// arguments, but it works to call it with three arguments when
// calling _cgo_panic.
//
//go:cgo_export_static crosscall2
//go:cgo_export_dynamic crosscall2

// Panic. The argument is converted into a Go string.

// Call like this in code compiled with gcc:
//   struct { const char *p; } a;
//   a.p = /* string to pass to panic */;
//   crosscall2(_cgo_panic, &a, sizeof a);
//   /* The function call will not return.  */

// TODO: We should export a regular C function to panic, change SWIG
// to use that instead of the above pattern, and then we can drop
// backwards-compatibility from crosscall2 and stop exporting it.

//go:linkname _runtime_cgo_panic_internal runtime._cgo_panic_internal
func _runtime_cgo_panic_internal(p *byte)

//go:linkname _cgo_panic _cgo_panic
//go:cgo_export_static _cgo_panic
//go:cgo_export_dynamic _cgo_panic
func _cgo_panic(a *struct{ cstr *byte }) {
	_runtime_cgo_panic_internal(a.cstr)
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

// Determines if the argc / argv passed to the library initialization functions
// are valid.
//go:cgo_import_static x_cgo_sys_lib_args_valid
//go:linkname x_cgo_sys_lib_args_valid x_cgo_sys_lib_args_valid
//go:linkname _cgo_sys_lib_args_valid _cgo_sys_lib_args_valid
var x_cgo_sys_lib_args_valid byte
var _cgo_sys_lib_args_valid = &x_cgo_sys_lib_args_valid

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

// Indicates whether a dummy thread key has been created or not.
//
// When calling go exported function from C, we register a destructor
// callback, for a dummy thread key, by using pthread_key_create.

//go:cgo_import_static x_cgo_pthread_key_created
//go:linkname x_cgo_pthread_key_created x_cgo_pthread_key_created
//go:linkname _cgo_pthread_key_created _cgo_pthread_key_created
var x_cgo_pthread_key_created byte
var _cgo_pthread_key_created = &x_cgo_pthread_key_created

// Export crosscall2 to a c function pointer variable.
// Used to dropm in pthread key destructor, while C thread is exiting.

//go:cgo_import_static x_crosscall2_ptr
//go:linkname x_crosscall2_ptr x_crosscall2_ptr
//go:linkname _crosscall2_ptr _crosscall2_ptr
var x_crosscall2_ptr byte
var _crosscall2_ptr = &x_crosscall2_ptr

// Set the x_crosscall2_ptr C function pointer variable point to crosscall2.
// It's for the runtime package to call at init time.
func set_crosscall2()

//go:linkname _set_crosscall2 runtime.set_crosscall2
var _set_crosscall2 = set_crosscall2

// Store the g into the thread-specific value.
// So that pthread_key_destructor will dropm when the thread is exiting.

//go:cgo_import_static x_cgo_bindm
//go:linkname x_cgo_bindm x_cgo_bindm
//go:linkname _cgo_bindm _cgo_bindm
var x_cgo_bindm byte
var _cgo_bindm = &x_cgo_bindm

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

// Calls a libc function to execute background work injected via libc
// interceptors, such as processing pending signals under the thread
// sanitizer.
//
// Left as a nil pointer if no libc interceptors are expected.

//go:cgo_import_static _cgo_yield
//go:linkname _cgo_yield _cgo_yield
var _cgo_yield unsafe.Pointer

//go:cgo_export_static _cgo_topofstack
//go:cgo_export_dynamic _cgo_topofstack

// x_cgo_getstackbound gets the thread's C stack size and
// set the G's stack bound based on the stack size.

//go:cgo_import_static x_cgo_getstackbound
//go:linkname x_cgo_getstackbound x_cgo_getstackbound
//go:linkname _cgo_getstackbound _cgo_getstackbound
var x_cgo_getstackbound byte
var _cgo_getstackbound = &x_cgo_getstackbound

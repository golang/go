// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !goexperiment.cgobindings

package cgo

import "unsafe"

// Calls a libc function to execute background work injected via libc
// interceptors, such as processing pending signals under the thread
// sanitizer.
//
// Left as a nil pointer if no libc interceptors are expected.

//go:cgo_import_static _cgo_yield
//go:linkname _cgo_yield _cgo_yield
var _cgo_yield unsafe.Pointer

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

// Sets the traceback, context, and symbolizer functions. See
// runtime.SetCgoTraceback.

//go:cgo_import_static x_cgo_set_traceback_functions
//go:linkname x_cgo_set_traceback_functions x_cgo_set_traceback_functions
//go:linkname _cgo_set_traceback_functions _cgo_set_traceback_functions
var x_cgo_set_traceback_functions byte
var _cgo_set_traceback_functions = &x_cgo_set_traceback_functions

// Call the traceback function registered with x_cgo_set_traceback_functions.

//go:cgo_import_static x_cgo_call_traceback_function
//go:linkname x_cgo_call_traceback_function x_cgo_call_traceback_function
//go:linkname _cgo_call_traceback_function _cgo_call_traceback_function
var x_cgo_call_traceback_function byte
var _cgo_call_traceback_function = &x_cgo_call_traceback_function

// Call the symbolizer function registered with x_cgo_set_symbolizer_functions.

//go:cgo_import_static x_cgo_call_symbolizer_function
//go:linkname x_cgo_call_symbolizer_function x_cgo_call_symbolizer_function
//go:linkname _cgo_call_symbolizer_function _cgo_call_symbolizer_function
var x_cgo_call_symbolizer_function byte
var _cgo_call_symbolizer_function = &x_cgo_call_symbolizer_function

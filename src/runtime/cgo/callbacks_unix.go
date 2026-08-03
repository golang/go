// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package cgo

import _ "unsafe" // for go:linkname

// We need to reference the addresses of the C functions in Go code,
// but we don't want to reference them directly in text. Otherwise it
// may cause dynamic relocations in the text segment, or even build
// failures in some settings, e.g. linking with -Bsymbolic-functions.
// Reference them via variables with an extra level of indirection to
// avoid that.

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
// x_cgo_thread_start except that it doesn't update any Go state.

//go:cgo_import_static x_cgo_sys_thread_create
//go:linkname x_cgo_sys_thread_create x_cgo_sys_thread_create
//go:linkname _cgo_sys_thread_create _cgo_sys_thread_create
var x_cgo_sys_thread_create byte
var _cgo_sys_thread_create = &x_cgo_sys_thread_create

// Store the g into the thread-specific value.
// So that pthread_key_destructor will dropm when the thread is exiting.

//go:cgo_import_static x_cgo_bindm
//go:linkname x_cgo_bindm x_cgo_bindm
//go:linkname _cgo_bindm _cgo_bindm
var x_cgo_bindm byte
var _cgo_bindm = &x_cgo_bindm

// x_cgo_getstackbound gets the thread's C stack size and
// set the G's stack bound based on the stack size.

//go:cgo_import_static x_cgo_getstackbound
//go:linkname x_cgo_getstackbound x_cgo_getstackbound
//go:linkname _cgo_getstackbound _cgo_getstackbound
var x_cgo_getstackbound byte
var _cgo_getstackbound = &x_cgo_getstackbound

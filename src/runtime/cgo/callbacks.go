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

// Set the x_crosscall2_ptr C function pointer variable point to crosscall2.
// It's for the runtime package to call at init time.
func set_crosscall2()

//go:linkname _set_crosscall2 runtime.set_crosscall2
var _set_crosscall2 = set_crosscall2

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

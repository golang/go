// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

//go:cgo_export_static main

// Filled in by runtime/cgo when linked into binary.

//go:linkname _cgo_init _cgo_init
//go:linkname _cgo_thread_start _cgo_thread_start
//go:linkname _cgo_sys_thread_create _cgo_sys_thread_create
//go:linkname _cgo_notify_runtime_init_done _cgo_notify_runtime_init_done
//go:linkname _cgo_callers _cgo_callers
//go:linkname _cgo_set_context_function _cgo_set_context_function
//go:linkname _cgo_yield _cgo_yield
//go:linkname _cgo_pthread_key_created _cgo_pthread_key_created
//go:linkname _cgo_bindm _cgo_bindm
//go:linkname _cgo_getstackbound _cgo_getstackbound

var (
	_cgo_init                     unsafe.Pointer
	_cgo_thread_start             unsafe.Pointer
	_cgo_sys_thread_create        unsafe.Pointer
	_cgo_notify_runtime_init_done unsafe.Pointer
	_cgo_callers                  unsafe.Pointer
	_cgo_set_context_function     unsafe.Pointer
	_cgo_yield                    unsafe.Pointer
	_cgo_pthread_key_created      unsafe.Pointer
	_cgo_bindm                    unsafe.Pointer
	_cgo_getstackbound            unsafe.Pointer
)

// iscgo is set to true by the runtime/cgo package
//
// iscgo should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ebitengine/purego
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname iscgo
var iscgo bool

// set_crosscall2 is set by the runtime/cgo package
// set_crosscall2 should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/ebitengine/purego
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname set_crosscall2
var set_crosscall2 func()

// cgoHasExtraM is set on startup when an extra M is created for cgo.
// The extra M must be created before any C/C++ code calls cgocallback.
var cgoHasExtraM bool

// cgoUse is called by cgo-generated code (using go:linkname to get at
// an unexported name). The calls serve two purposes:
// 1) they are opaque to escape analysis, so the argument is considered to
// escape to the heap.
// 2) they keep the argument alive until the call site; the call is emitted after
// the end of the (presumed) use of the argument by C.
// cgoUse should not actually be called (see cgoAlwaysFalse).
func cgoUse(any) { throw("cgoUse should not be called") }

// cgoKeepAlive is called by cgo-generated code (using go:linkname to get at
// an unexported name). Unlike cgoUse The calls serve one purposes:
// 1) they keep the argument alive until the call site; the call is emitted after
// the end of the (presumed) use of the argument by C.
// cgoKeepAlive should not actually be called (see cgoAlwaysFalse).
func cgoKeepAlive(any) { throw("cgoKeepAlive should not be called") }

// cgoAlwaysFalse is a boolean value that is always false.
// The cgo-generated code says if cgoAlwaysFalse { cgoUse(p) },
// or if cgoAlwaysFalse { cgoUse(p) }.
// The compiler cannot see that cgoAlwaysFalse is always false,
// so it emits the test and keeps the call, giving the desired
// escape/alive analysis result. The test is cheaper than the call.
var cgoAlwaysFalse bool

var cgo_yield = &_cgo_yield

func cgoNoCallback(v bool) {
	g := getg()
	if g.nocgocallback && v {
		panic("runtime: unexpected setting cgoNoCallback")
	}
	g.nocgocallback = v
}

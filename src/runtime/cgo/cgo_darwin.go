// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && darwin

package cgo

const (
	_SIG_SETMASK             = 3
	_EAGAIN                  = 35
	_PTHREAD_CREATE_DETACHED = 2
)

//go:cgo_import_dynamic _cgo_libc_setenv setenv "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_unsetenv unsetenv "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setstacksize pthread_attr_setstacksize "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_self pthread_self "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_get_stackaddr_np pthread_get_stackaddr_np "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_get_stacksize_np pthread_get_stacksize_np "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask pthread_sigmask "/usr/lib/libSystem.B.dylib"
//go:cgo_import_dynamic _cgo_libc_sigfillset sigfillset "/usr/lib/libSystem.B.dylib"

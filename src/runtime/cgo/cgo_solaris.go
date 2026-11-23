// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && solaris

package cgo

const (
	_SIG_SETMASK             = 3
	_EAGAIN                  = 11
	_PTHREAD_CREATE_DETACHED = 0x40
)

//go:cgo_import_dynamic _ _ "libc.so"
//go:cgo_import_dynamic _cgo_libc_setenv setenv "libc.so"
//go:cgo_import_dynamic _cgo_libc_unsetenv unsetenv "libc.so"
//go:cgo_import_dynamic _cgo_libc_sigfillset sigfillset "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstacksize pthread_attr_getstacksize "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setstacksize pthread_attr_setstacksize "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstack pthread_attr_getstack "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_self pthread_self "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask pthread_sigmask "libc.so"

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && freebsd

package cgo

const (
	_SIG_SETMASK             = 3
	_EAGAIN                  = 35
	_PTHREAD_CREATE_DETACHED = 1
)

//go:cgo_ldflag "-lpthread"

//go:cgo_import_dynamic _ _ "libc.so.7"
//go:cgo_import_dynamic _ _ "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_setenv setenv "libc.so.7"
//go:cgo_import_dynamic _cgo_libc_unsetenv unsetenv "libc.so.7"
//go:cgo_import_dynamic _cgo_libc_sigfillset sigfillset "libc.so.7"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstacksize pthread_attr_getstacksize "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "libthr.so.3"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask pthread_sigmask "libthr.so.3"

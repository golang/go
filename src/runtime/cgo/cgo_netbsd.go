// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && netbsd

package cgo

const (
	_SIG_SETMASK             = 3
	_EAGAIN                  = 35
	_PTHREAD_CREATE_DETACHED = 1
)

//go:cgo_ldflag "-lpthread"

//go:cgo_import_dynamic _ _ "libc.so.12"
//go:cgo_import_dynamic _ _ "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_setenv setenv "libc.so.12"
//go:cgo_import_dynamic _cgo_libc_unsetenv __unsetenv13 "libc.so.12"
//go:cgo_import_dynamic _cgo_libc_sigfillset __sigfillset14 "libc.so.12"
//go:cgo_import_dynamic _cgo_libc_sigaltstack __sigaltstack14 "libc.so.12"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "libpthread.so.1"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask pthread_sigmask "libpthread.so.1"

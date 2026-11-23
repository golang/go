// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && openbsd

package cgo

const (
	_SIG_SETMASK             = 3
	_EAGAIN                  = 35
	_PTHREAD_CREATE_DETACHED = 1
)

//go:cgo_ldflag "-lpthread"

// Some of these functions are also imported by the runtime under libc_*
// names (see runtime/sys_openbsd.go). The runtime/cgo imports use separate
// _cgo_libc_* symbols to avoid colliding with the runtime's libc_* symbols,
// so the dynamic symbol table may contain two undefined entries for one
// function.
// This is valid ELF, as each relocation names its own entry.

//go:cgo_import_dynamic _ _ "libc.so"
//go:cgo_import_dynamic _ _ "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_setenv setenv "libc.so"
//go:cgo_import_dynamic _cgo_libc_unsetenv unsetenv "libc.so"
//go:cgo_import_dynamic _cgo_libc_sigfillset sigfillset "libc.so"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "libpthread.so"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask pthread_sigmask "libpthread.so"

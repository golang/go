// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && aix

package cgo

import "unsafe"

const (
	_SIG_SETMASK             = 2
	_EAGAIN                  = 11
	_PTHREAD_CREATE_DETACHED = 1
)

// On AIX, C function pointers are 24-byte XCOFF function descriptors
// defined via DEFINE_PPC64X_FUNCDESC in asm_unix_ppc64x.s.
var cgoCallTrampoline uintptr
var pthreadKeyDestructor uintptr
var x_cgo_threadentry uintptr

//go:nosplit
//go:norace
func cgoCallTrampolinePC() unsafe.Pointer {
	return unsafe.Pointer(&cgoCallTrampoline)
}

//go:nosplit
//go:norace
func pthreadKeyDestructorPC() uintptr {
	return uintptr(unsafe.Pointer(&pthreadKeyDestructor))
}

//go:nosplit
//go:norace
func threadentryPC() uintptr {
	return uintptr(unsafe.Pointer(&x_cgo_threadentry))
}

//go:cgo_export_static _GLOBAL__I_65535_0_runtime_cgo

//go:cgo_ldflag "-Wl,-berok"

//go:cgo_import_dynamic _cgo_libc_setenv setenv "libc.a/shr_64.o"
//go:cgo_import_dynamic _cgo_libc_unsetenv unsetenv "libc.a/shr_64.o"
//go:cgo_import_dynamic _cgo_libc_sigfillset sigfillset "libc.a/shr_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "libpthread.a/shr_xpg5_64.o"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask sigthreadmask "libpthread.a/shr_xpg5_64.o"

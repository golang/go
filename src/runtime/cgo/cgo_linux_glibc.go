// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.runtimecgopurego && !msan && !tsan && linux && !android && (386 || amd64 || arm || arm64 || loong64 || riscv64)

package cgo

import "unsafe"

// cSigaction matches libc's struct sigaction on Linux (glibc and musl) across
// 32-bit and 64-bit architectures supporting cgoSigaction.
type cSigaction struct {
	handler  uintptr
	mask     [128 / unsafe.Sizeof(uintptr(0))]uintptr // sigset_t
	flags    int32
	restorer uintptr
}

//go:cgo_ldflag "-pthread"

// Specify "libc.so.6" so internal linking without C host objects emits a
// DT_NEEDED entry for glibc's dynamic loader; external linking ignores the
// library field, !android excludes bionic, and musl treats any "libc.so.*"
// SONAME as libc itself.
//go:cgo_import_dynamic _ _ "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_mmap mmap "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_munmap munmap "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_sigaction sigaction "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_sigfillset sigfillset "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_errno_location __errno_location "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setenv setenv "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_unsetenv unsetenv "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_clearenv clearenv "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_pthread_create pthread_create "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_init pthread_attr_init "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_setdetachstate pthread_attr_setdetachstate "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstacksize pthread_attr_getstacksize "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_destroy pthread_attr_destroy "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_key_create pthread_key_create "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_setspecific pthread_setspecific "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_self pthread_self "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_getattr_np pthread_getattr_np "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_attr_getstack pthread_attr_getstack "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_pthread_sigmask pthread_sigmask "libpthread.so.0"
//go:cgo_import_dynamic _cgo_libc_setegid setegid "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_seteuid seteuid "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setgid setgid "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setgroups setgroups "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setregid setregid "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setresgid setresgid "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setresuid setresuid "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setreuid setreuid "libc.so.6"
//go:cgo_import_dynamic _cgo_libc_setuid setuid "libc.so.6"

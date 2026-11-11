// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && race && !cgo && freebsd && amd64

package race

//go:cgo_import_dynamic __error __error
//go:cgo_import_dynamic __syscall __syscall
//go:cgo_import_dynamic _umtx_op _umtx_op
//go:cgo_import_dynamic abort abort
//go:cgo_import_dynamic clock_gettime clock_gettime
//go:cgo_import_dynamic dl_iterate_phdr dl_iterate_phdr
//go:cgo_import_dynamic dlinfo dlinfo
//go:cgo_import_dynamic exit exit
//go:cgo_import_dynamic getentropy getentropy
//go:cgo_import_dynamic getpid getpid
//go:cgo_import_dynamic getrlimit getrlimit
//go:cgo_import_dynamic getrusage getrusage
//go:cgo_import_dynamic getuid getuid
//go:cgo_import_dynamic isatty isatty
//go:cgo_import_dynamic mkdir mkdir
//go:cgo_import_dynamic pipe pipe
//go:cgo_import_dynamic pthread_attr_destroy pthread_attr_destroy
//go:cgo_import_dynamic pthread_attr_get_np pthread_attr_get_np
//go:cgo_import_dynamic pthread_attr_getstack pthread_attr_getstack
//go:cgo_import_dynamic pthread_attr_init pthread_attr_init
//go:cgo_import_dynamic pthread_self pthread_self
//go:cgo_import_dynamic setrlimit setrlimit
//go:cgo_import_dynamic sigaction sigaction
//go:cgo_import_dynamic sigdelset sigdelset
//go:cgo_import_dynamic sigismember sigismember
//go:cgo_import_dynamic sysconf sysconf
//go:cgo_import_dynamic thr_self thr_self

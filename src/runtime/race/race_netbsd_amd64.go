// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && race && !cgo && netbsd && amd64

package race

//go:cgo_import_dynamic __clock_gettime50 __clock_gettime50
//go:cgo_import_dynamic __errno __errno
//go:cgo_import_dynamic __fork __fork
//go:cgo_import_dynamic __ftruncate __ftruncate
//go:cgo_import_dynamic __getrusage50 __getrusage50
//go:cgo_import_dynamic __libc_thr_self __libc_thr_self
//go:cgo_import_dynamic __libc_thr_yield __libc_thr_yield
//go:cgo_import_dynamic __lseek __lseek
//go:cgo_import_dynamic __mmap __mmap
//go:cgo_import_dynamic __sigaction14 __sigaction14
//go:cgo_import_dynamic __sysctl __sysctl
//go:cgo_import_dynamic _sys___nanosleep50 _sys___nanosleep50
//go:cgo_import_dynamic _sys___sigprocmask14 _sys___sigprocmask14
//go:cgo_import_dynamic _sys___wait450 _sys___wait450
//go:cgo_import_dynamic _sys_close _sys_close
//go:cgo_import_dynamic _sys_execve _sys_execve
//go:cgo_import_dynamic _sys_open _sys_open
//go:cgo_import_dynamic _sys_read _sys_read
//go:cgo_import_dynamic _sys_readlink _sys_readlink
//go:cgo_import_dynamic _sys_sched_yield _sys_sched_yield
//go:cgo_import_dynamic _sys_write _sys_write
//go:cgo_import_dynamic abort abort
//go:cgo_import_dynamic dl_iterate_phdr dl_iterate_phdr
//go:cgo_import_dynamic dlsym dlsym
//go:cgo_import_dynamic exit exit
//go:cgo_import_dynamic getpid getpid
//go:cgo_import_dynamic getrlimit getrlimit
//go:cgo_import_dynamic getuid getuid
//go:cgo_import_dynamic isatty isatty
//go:cgo_import_dynamic mkdir mkdir
//go:cgo_import_dynamic pipe pipe
//go:cgo_import_dynamic pthread_attr_destroy pthread_attr_destroy
//go:cgo_import_dynamic pthread_attr_getstack pthread_attr_getstack
//go:cgo_import_dynamic pthread_attr_init pthread_attr_init
//go:cgo_import_dynamic pthread_getattr_np pthread_getattr_np
//go:cgo_import_dynamic setrlimit setrlimit
//go:cgo_import_dynamic sysconf sysconf

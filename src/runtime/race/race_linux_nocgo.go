// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.cgobindings && race && !cgo && linux && (amd64 || arm64 || loong64 || ppc64le || riscv64 || s390x)

package race

//go:cgo_import_dynamic __errno_location __errno_location
//go:cgo_import_dynamic __sched_cpucount __sched_cpucount
//go:cgo_import_dynamic __stack_chk_fail __stack_chk_fail
//go:cgo_import_dynamic __syslog_chk __syslog_chk
//go:cgo_import_dynamic abort abort
//go:cgo_import_dynamic clock_gettime clock_gettime
//go:cgo_import_dynamic dl_iterate_phdr dl_iterate_phdr
//go:cgo_import_dynamic exit exit
//go:cgo_import_dynamic fcntl fcntl
//go:cgo_import_dynamic ffs ffs
//go:cgo_import_dynamic getrlimit getrlimit
//go:cgo_import_dynamic getrusage getrusage
//go:cgo_import_dynamic getuid getuid
//go:cgo_import_dynamic isatty isatty
//go:cgo_import_dynamic mkdir mkdir
//go:cgo_import_dynamic personality personality
//go:cgo_import_dynamic pipe pipe
//go:cgo_import_dynamic pthread_attr_destroy pthread_attr_destroy
//go:cgo_import_dynamic pthread_attr_getstack pthread_attr_getstack
//go:cgo_import_dynamic pthread_attr_init pthread_attr_init
//go:cgo_import_dynamic pthread_getattr_np pthread_getattr_np
//go:cgo_import_dynamic pthread_self pthread_self
//go:cgo_import_dynamic sched_getaffinity sched_getaffinity
//go:cgo_import_dynamic sched_yield sched_yield
//go:cgo_import_dynamic setrlimit setrlimit
//go:cgo_import_dynamic sigaction sigaction
//go:cgo_import_dynamic strlen strlen
//go:cgo_import_dynamic syscall syscall
//go:cgo_import_dynamic sysconf sysconf
//go:cgo_import_dynamic syslog syslog

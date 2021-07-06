// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe" // for go:linkname

//go:cgo_import_dynamic libc_chdir chdir "libc.so"
//go:cgo_import_dynamic libc_chroot chroot "libc.so"
//go:cgo_import_dynamic libc_close close "libc.so"
//go:cgo_import_dynamic libc_execve execve "libc.so"
//go:cgo_import_dynamic libc_fcntl fcntl "libc.so"
//go:cgo_import_dynamic libc_forkx forkx "libc.so"
//go:cgo_import_dynamic libc_gethostname gethostname "libc.so"
//go:cgo_import_dynamic libc_getpid getpid "libc.so"
//go:cgo_import_dynamic libc_ioctl ioctl "libc.so"
//go:cgo_import_dynamic libc_setgid setgid "libc.so"
//go:cgo_import_dynamic libc_setgroups setgroups "libc.so"
//go:cgo_import_dynamic libc_setsid setsid "libc.so"
//go:cgo_import_dynamic libc_setuid setuid "libc.so"
//go:cgo_import_dynamic libc_setpgid setpgid "libc.so"
//go:cgo_import_dynamic libc_syscall syscall "libc.so"
//go:cgo_import_dynamic libc_wait4 wait4 "libc.so"

//go:linkname libc_chdir libc_chdir
//go:linkname libc_chroot libc_chroot
//go:linkname libc_close libc_close
//go:linkname libc_execve libc_execve
//go:linkname libc_fcntl libc_fcntl
//go:linkname libc_forkx libc_forkx
//go:linkname libc_gethostname libc_gethostname
//go:linkname libc_getpid libc_getpid
//go:linkname libc_ioctl libc_ioctl
//go:linkname libc_setgid libc_setgid
//go:linkname libc_setgroups libc_setgroups
//go:linkname libc_setsid libc_setsid
//go:linkname libc_setuid libc_setuid
//go:linkname libc_setpgid libc_setpgid
//go:linkname libc_syscall libc_syscall
//go:linkname libc_wait4 libc_wait4

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package os

import "syscall"

// Commonly known Unix errors.
var (
	EPERM        error = syscall.EPERM
	ENOENT       error = syscall.ENOENT
	ESRCH        error = syscall.ESRCH
	EINTR        error = syscall.EINTR
	EIO          error = syscall.EIO
	E2BIG        error = syscall.E2BIG
	ENOEXEC      error = syscall.ENOEXEC
	EBADF        error = syscall.EBADF
	ECHILD       error = syscall.ECHILD
	EDEADLK      error = syscall.EDEADLK
	ENOMEM       error = syscall.ENOMEM
	EACCES       error = syscall.EACCES
	EFAULT       error = syscall.EFAULT
	EBUSY        error = syscall.EBUSY
	EEXIST       error = syscall.EEXIST
	EXDEV        error = syscall.EXDEV
	ENODEV       error = syscall.ENODEV
	ENOTDIR      error = syscall.ENOTDIR
	EISDIR       error = syscall.EISDIR
	EINVAL       error = syscall.EINVAL
	ENFILE       error = syscall.ENFILE
	EMFILE       error = syscall.EMFILE
	ENOTTY       error = syscall.ENOTTY
	EFBIG        error = syscall.EFBIG
	ENOSPC       error = syscall.ENOSPC
	ESPIPE       error = syscall.ESPIPE
	EROFS        error = syscall.EROFS
	EMLINK       error = syscall.EMLINK
	EPIPE        error = syscall.EPIPE
	EAGAIN       error = syscall.EAGAIN
	EDOM         error = syscall.EDOM
	ERANGE       error = syscall.ERANGE
	EADDRINUSE   error = syscall.EADDRINUSE
	ECONNREFUSED error = syscall.ECONNREFUSED
	ENAMETOOLONG error = syscall.ENAMETOOLONG
	EAFNOSUPPORT error = syscall.EAFNOSUPPORT
	ETIMEDOUT    error = syscall.ETIMEDOUT
	ENOTCONN     error = syscall.ENOTCONN
)

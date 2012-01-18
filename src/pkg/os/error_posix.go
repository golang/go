// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package os

import syscall "syscall"

// Commonly known Unix errors.
var (
	EPERM        error = syscall.EPERM
	ENOENT       error = syscall.ENOENT
	ESRCH        error = syscall.ESRCH
	EINTR        error = syscall.EINTR
	EIO          error = syscall.EIO
	ENXIO        error = syscall.ENXIO
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

// SyscallError records an error from a specific system call.
type SyscallError struct {
	Syscall string
	Errno   error
}

func (e *SyscallError) Error() string { return e.Syscall + ": " + e.Errno.Error() }

// NewSyscallError returns, as an error, a new SyscallError
// with the given system call name and error details.
// As a convenience, if err is nil, NewSyscallError returns nil.
func NewSyscallError(syscall string, err error) error {
	if err == nil {
		return nil
	}
	return &SyscallError{syscall, err}
}

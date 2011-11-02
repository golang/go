// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd windows

package os

import syscall "syscall"

// Errno is the Unix error number.  Names such as EINVAL are simple
// wrappers to convert the error number into an error.
type Errno int64

func (e Errno) Error() string { return syscall.Errstr(int(e)) }

func (e Errno) Temporary() bool {
	return e == Errno(syscall.EINTR) || e == Errno(syscall.EMFILE) || e.Timeout()
}

func (e Errno) Timeout() bool {
	return e == Errno(syscall.EAGAIN) || e == Errno(syscall.EWOULDBLOCK) || e == Errno(syscall.ETIMEDOUT)
}

// Commonly known Unix errors.
var (
	EPERM        error = Errno(syscall.EPERM)
	ENOENT       error = Errno(syscall.ENOENT)
	ESRCH        error = Errno(syscall.ESRCH)
	EINTR        error = Errno(syscall.EINTR)
	EIO          error = Errno(syscall.EIO)
	ENXIO        error = Errno(syscall.ENXIO)
	E2BIG        error = Errno(syscall.E2BIG)
	ENOEXEC      error = Errno(syscall.ENOEXEC)
	EBADF        error = Errno(syscall.EBADF)
	ECHILD       error = Errno(syscall.ECHILD)
	EDEADLK      error = Errno(syscall.EDEADLK)
	ENOMEM       error = Errno(syscall.ENOMEM)
	EACCES       error = Errno(syscall.EACCES)
	EFAULT       error = Errno(syscall.EFAULT)
	EBUSY        error = Errno(syscall.EBUSY)
	EEXIST       error = Errno(syscall.EEXIST)
	EXDEV        error = Errno(syscall.EXDEV)
	ENODEV       error = Errno(syscall.ENODEV)
	ENOTDIR      error = Errno(syscall.ENOTDIR)
	EISDIR       error = Errno(syscall.EISDIR)
	EINVAL       error = Errno(syscall.EINVAL)
	ENFILE       error = Errno(syscall.ENFILE)
	EMFILE       error = Errno(syscall.EMFILE)
	ENOTTY       error = Errno(syscall.ENOTTY)
	EFBIG        error = Errno(syscall.EFBIG)
	ENOSPC       error = Errno(syscall.ENOSPC)
	ESPIPE       error = Errno(syscall.ESPIPE)
	EROFS        error = Errno(syscall.EROFS)
	EMLINK       error = Errno(syscall.EMLINK)
	EPIPE        error = Errno(syscall.EPIPE)
	EAGAIN       error = Errno(syscall.EAGAIN)
	EDOM         error = Errno(syscall.EDOM)
	ERANGE       error = Errno(syscall.ERANGE)
	EADDRINUSE   error = Errno(syscall.EADDRINUSE)
	ECONNREFUSED error = Errno(syscall.ECONNREFUSED)
	ENAMETOOLONG error = Errno(syscall.ENAMETOOLONG)
	EAFNOSUPPORT error = Errno(syscall.EAFNOSUPPORT)
	ETIMEDOUT    error = Errno(syscall.ETIMEDOUT)
	ENOTCONN     error = Errno(syscall.ENOTCONN)
)

// SyscallError records an error from a specific system call.
type SyscallError struct {
	Syscall string
	Errno   Errno
}

func (e *SyscallError) Error() string { return e.Syscall + ": " + e.Errno.Error() }

// Note: If the name of the function NewSyscallError changes,
// pkg/go/doc/doc.go should be adjusted since it hardwires
// this name in a heuristic.

// NewSyscallError returns, as an error, a new SyscallError
// with the given system call name and error details.
// As a convenience, if errno is 0, NewSyscallError returns nil.
func NewSyscallError(syscall string, errno int) error {
	if errno == 0 {
		return nil
	}
	return &SyscallError{syscall, Errno(errno)}
}

func iserror(errno int) bool {
	return errno != 0
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"

// An Error can represent any printable error condition.
type Error interface {
	String() string
}

// A helper type that can be embedded or wrapped to simplify satisfying
// Error.
type ErrorString string
func (e ErrorString) String() string {
	return string(e)
}

// NewError converts s to an ErrorString, which satisfies the Error interface.
func NewError(s string) Error {
	return ErrorString(s)
}

// Errno is the Unix error number.  Names such as EINVAL are simple
// wrappers to convert the error number into an Error.
type Errno int64
func (e Errno) String() string {
	return syscall.Errstr(int(e))
}

// ErrnoToError converts errno to an Error (underneath, an Errno).
// It returns nil for the "no error" errno.
func ErrnoToError(errno int) Error {
	if errno == 0 {
		return nil
	}
	return Errno(errno)
}

// Commonly known Unix errors.
var (
	EPERM Error = Errno(syscall.EPERM);
	ENOENT Error = Errno(syscall.ENOENT);
	ESRCH Error = Errno(syscall.ESRCH);
	EINTR Error = Errno(syscall.EINTR);
	EIO Error = Errno(syscall.EIO);
	ENXIO Error = Errno(syscall.ENXIO);
	E2BIG Error = Errno(syscall.E2BIG);
	ENOEXEC Error = Errno(syscall.ENOEXEC);
	EBADF Error = Errno(syscall.EBADF);
	ECHILD Error = Errno(syscall.ECHILD);
	EDEADLK Error = Errno(syscall.EDEADLK);
	ENOMEM Error = Errno(syscall.ENOMEM);
	EACCES Error = Errno(syscall.EACCES);
	EFAULT Error = Errno(syscall.EFAULT);
	ENOTBLK Error = Errno(syscall.ENOTBLK);
	EBUSY Error = Errno(syscall.EBUSY);
	EEXIST Error = Errno(syscall.EEXIST);
	EXDEV Error = Errno(syscall.EXDEV);
	ENODEV Error = Errno(syscall.ENODEV);
	ENOTDIR Error = Errno(syscall.ENOTDIR);
	EISDIR Error = Errno(syscall.EISDIR);
	EINVAL Error = Errno(syscall.EINVAL);
	ENFILE Error = Errno(syscall.ENFILE);
	EMFILE Error = Errno(syscall.EMFILE);
	ENOTTY Error = Errno(syscall.ENOTTY);
	ETXTBSY Error = Errno(syscall.ETXTBSY);
	EFBIG Error = Errno(syscall.EFBIG);
	ENOSPC Error = Errno(syscall.ENOSPC);
	ESPIPE Error = Errno(syscall.ESPIPE);
	EROFS Error = Errno(syscall.EROFS);
	EMLINK Error = Errno(syscall.EMLINK);
	EPIPE Error = Errno(syscall.EPIPE);
	EAGAIN Error = Errno(syscall.EAGAIN);
	EDOM Error = Errno(syscall.EDOM);
	ERANGE Error = Errno(syscall.ERANGE);
	EADDRINUSE Error = Errno(syscall.EADDRINUSE);
	ECONNREFUSED Error = Errno(syscall.ECONNREFUSED);
	ENAMETOOLONG Error = Errno(syscall.ENAMETOOLONG);
)


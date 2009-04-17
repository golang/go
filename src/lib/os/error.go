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
func (e *ErrorString) String() string {
	return *e
}

// Errno is the Unix error number.  Names such as EINVAL are simple
// wrappers to convert the error number into an Error.
type Errno int64
func (e Errno) String() string {
	return syscall.Errstr(e)
}

// ErrnoToError calls NewError to create an _Error object for the string
// associated with Unix error code errno.
func ErrnoToError(errno int64) Error {
	if errno == 0 {
		return nil
	}
	return Errno(errno)
}

// Commonly known Unix errors.
var (
	ENONE Error = Errno(syscall.ENONE);
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
)

// -----------------------
// Everything below here is deprecated.
// Delete when all callers of NewError are gone and their uses converted
// to the new error scheme (for an example, see template).

// _Error is a structure wrapping a string describing an error.
// Errors are singleton structures, created by NewError, so their addresses can
// be compared to test for equality. A nil Error pointer means ``no error''.
// Use the String() method to get the contents; it handles the nil case.
// The Error type is intended for use by any package that wishes to define
// error strings.
type _Error struct {
	s string
}

// Table of all known errors in system.  Use the same error string twice,
// get the same *os._Error.
var errorStringTab = make(map[string] Error);

// These functions contain a race if two goroutines add identical
// errors simultaneously but the consequences are unimportant.

// NewError allocates an Error object, but if s has been seen before,
// shares the _Error associated with that message.
func NewError(s string) Error {
	if s == "" {
		return nil
	}
	err, ok := errorStringTab[s];
	if ok {
		return err
	}
	err = &_Error{s};
	errorStringTab[s] = err;
	return err;
}


// String returns the string associated with the _Error.
func (e *_Error) String() string {
	if e == nil {
		return "No _Error"
	}
	return e.s
}

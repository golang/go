// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"

// Errors are singleton structures. Use the String() method to get their contents --
// it handles the nil (no error) case.
type Error struct {
	s string
}

// Indexed by errno.
// If we worry about syscall speed (only relevant on failure), we could
// make it an array, but it's probably not important.
var errorTab = make(map[int64] *Error);

// Table of all known errors in system.  Use the same error string twice,
// get the same *os.Error.
var errorStringTab = make(map[string] *Error);

// These functions contain a race if two goroutines add identical
// errors simultaneously but the consequences are unimportant.

// Allocate an Error object, but if it's been seen before, share that one.
func NewError(s string) *Error {
	if s == "" {
		return nil
	}
	err, ok := errorStringTab[s];
	if ok {
		return err
	}
	err = &Error{s};
	errorStringTab[s] = err;
	return err;
}

// Allocate an Error objecct, but if it's been seen before, share that one.
func ErrnoToError(errno int64) *Error {
	if errno == 0 {
		return nil
	}
	// Quick lookup by errno.
	err, ok := errorTab[errno];
	if ok {
		return err
	}
	err = NewError(syscall.Errstr(errno));
	errorTab[errno] = err;
	return err;
}

var (
	ENONE = ErrnoToError(syscall.ENONE);
	EPERM = ErrnoToError(syscall.EPERM);
	ENOENT = ErrnoToError(syscall.ENOENT);
	ESRCH = ErrnoToError(syscall.ESRCH);
	EINTR = ErrnoToError(syscall.EINTR);
	EIO = ErrnoToError(syscall.EIO);
	ENXIO = ErrnoToError(syscall.ENXIO);
	E2BIG = ErrnoToError(syscall.E2BIG);
	ENOEXEC = ErrnoToError(syscall.ENOEXEC);
	EBADF = ErrnoToError(syscall.EBADF);
	ECHILD = ErrnoToError(syscall.ECHILD);
	EDEADLK = ErrnoToError(syscall.EDEADLK);
	ENOMEM = ErrnoToError(syscall.ENOMEM);
	EACCES = ErrnoToError(syscall.EACCES);
	EFAULT = ErrnoToError(syscall.EFAULT);
	ENOTBLK = ErrnoToError(syscall.ENOTBLK);
	EBUSY = ErrnoToError(syscall.EBUSY);
	EEXIST = ErrnoToError(syscall.EEXIST);
	EXDEV = ErrnoToError(syscall.EXDEV);
	ENODEV = ErrnoToError(syscall.ENODEV);
	ENOTDIR = ErrnoToError(syscall.ENOTDIR);
	EISDIR = ErrnoToError(syscall.EISDIR);
	EINVAL = ErrnoToError(syscall.EINVAL);
	ENFILE = ErrnoToError(syscall.ENFILE);
	EMFILE = ErrnoToError(syscall.EMFILE);
	ENOTTY = ErrnoToError(syscall.ENOTTY);
	ETXTBSY = ErrnoToError(syscall.ETXTBSY);
	EFBIG = ErrnoToError(syscall.EFBIG);
	ENOSPC = ErrnoToError(syscall.ENOSPC);
	ESPIPE = ErrnoToError(syscall.ESPIPE);
	EROFS = ErrnoToError(syscall.EROFS);
	EMLINK = ErrnoToError(syscall.EMLINK);
	EPIPE = ErrnoToError(syscall.EPIPE);
	EDOM = ErrnoToError(syscall.EDOM);
	ERANGE = ErrnoToError(syscall.ERANGE);
	EAGAIN = ErrnoToError(syscall.EAGAIN);
)

func (e *Error) String() string {
	if e == nil {
		return "No Error"
	}
	return e.s
}

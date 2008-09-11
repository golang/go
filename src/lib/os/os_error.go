// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"

// Errors are singleton structures. Use the String() method to get their contents --
// it handles the nil (no error) case.
export type Error struct {
	s string
}

var ErrorTab = new(map[int64] *Error);

export func ErrnoToError(errno int64) *Error {
	if errno == 0 {
		return nil
	}
	err, ok := ErrorTab[errno]
	if ok {
		return err
	}
	e := new(Error);
	e.s = syscall.errstr(errno);
	ErrorTab[errno] = e;
	return e;
}
export var (
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
const NoError = "No Error"

func (e *Error) String() string {
	if e == nil {
		return NoError
	}
	return e.s
}

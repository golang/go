// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"

// Support types and routines for OS library

func WriteString(fd int64, s string) (ret int64, err *Error);

export func StringToBytes(b *[]byte, s string) bool {
	if len(s) >= len(b) {
		return false
	}
	for i := 0; i < len(s); i++ {
		b[i] = s[i]
	}
	b[len(s)] = '\000';	// not necessary - memory is zeroed - but be explicit
	return true
}

// Errors are singleton structures. Use the Print()/String() methods to get their contents --
// it handles the nil (no error) case.

export type Error struct {
	s string
}

const NoError = "No Error"

func (e *Error) Print() {
	if e == nil {
		WriteString(2, NoError)
	} else {
		WriteString(2, e.s)
	}
}

func (e *Error) String() string {
	if e == nil {
		return NoError
	} else {
		return e.s
	}
}

var ErrorTab = new(map[int64] *Error);

func ErrnoToError(errno int64) *Error {
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

export func Open(name string, mode int64, flags int64) (ret int64, err *Error) {
	var buf [512]byte;
	if !StringToBytes(&buf, name) {
		return -1, ErrnoToError(syscall.ENAMETOOLONG)
	}
	r, e := syscall.open(&buf[0], mode, flags);
	return r, ErrnoToError(e)
}

export func Close(fd int64) (ret int64, err *Error) {
	r, e := syscall.close(fd);
	return r, ErrnoToError(e)
}

export func Read(fd int64, b *[]byte) (ret int64, err *Error) {
	r, e := syscall.read(fd, &b[0], int64(len(b)));
	return r, ErrnoToError(e)
}

export func Write(fd int64, b *[]byte) (ret int64, err *Error) {
	r, e := syscall.write(fd, &b[0], int64(len(b)));
	return r, ErrnoToError(e)
}

export func WriteString(fd int64, s string) (ret int64, err *Error) {
	b := new([]byte, len(s)+1);
	if !StringToBytes(b, s) {
		return -1, ErrnoToError(syscall.ENAMETOOLONG)
	}
	r, e := syscall.write(fd, &b[0], int64(len(s)));
	return r, ErrnoToError(e)
}


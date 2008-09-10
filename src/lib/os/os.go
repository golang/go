// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"

// Support types and routines for OS library

// FDs are wrappers for file descriptors
export type FD struct {
	fd int64
}

// Errors are singleton structures. Use the Print()/String() methods to get their contents --
// they handle the nil (no error) case.
export type Error struct {
	s string
}

export func NewFD(fd int64) *FD {
	if fd < 0 {
		return nil
	}
	n := new(FD);	// TODO(r): how about return &FD{fd} ?
	n.fd = fd;
	return n;
}

export var (
	Stdin = NewFD(0);
	Stdout = NewFD(1);
	Stderr = NewFD(2);
)

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

export func Open(name string, mode int64, flags int64) (fd *FD, err *Error) {
	var buf [512]byte;
	if !StringToBytes(&buf, name) {
		return nil, EINVAL
	}
	r, e := syscall.open(&buf[0], mode, flags);
	return NewFD(r), ErrnoToError(e)
}

func (fd *FD) Close() *Error {
	if fd == nil {
		return EINVAL
	}
	r, e := syscall.close(fd.fd);
	fd.fd = -1;  // so it can't be closed again
	return ErrnoToError(e)
}

func (fd *FD) Read(b *[]byte) (ret int64, err *Error) {
	if fd == nil {
		return -1, EINVAL
	}
	r, e := syscall.read(fd.fd, &b[0], int64(len(b)));
	return r, ErrnoToError(e)
}

func (fd *FD) Write(b *[]byte) (ret int64, err *Error) {
	if fd == nil {
		return -1, EINVAL
	}
	r, e := syscall.write(fd.fd, &b[0], int64(len(b)));
	return r, ErrnoToError(e)
}

func (fd *FD) WriteString(s string) (ret int64, err *Error) {
	if fd == nil {
		return -1, EINVAL
	}
	b := new([]byte, len(s)+1);
	if !StringToBytes(b, s) {
		return -1, EINVAL
	}
	r, e := syscall.write(fd.fd, &b[0], int64(len(s)));
	return r, ErrnoToError(e)
}

const NoError = "No Error"

func (e *Error) String() string {
	if e == nil {
		return NoError
	} else {
		return e.s
	}
}

func (e *Error) Print() {
	Stderr.WriteString(e.String())
}

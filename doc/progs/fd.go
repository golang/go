// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fd

import (
	"os";
	"syscall";
)

type FD struct {
	fildes  int64;  // file descriptor number
	name    string; // file name at Open time
}

func newFD(fd int64, name string) *FD {
	if fd < 0 {
		return nil
	}
	return &FD(fd, name)
}

var (
	Stdin  = newFD(0, "/dev/stdin");
	Stdout = newFD(1, "/dev/stdout");
	Stderr = newFD(2, "/dev/stderr");
)

func Open(name string, mode int64, perm int64) (fd *FD, err *os.Error) {
	r, e := syscall.Open(name, mode, perm);
	return newFD(r, name), os.ErrnoToError(e)
}

func (fd *FD) Close() *os.Error {
	if fd == nil {
		return os.EINVAL
	}
	r, e := syscall.Close(fd.fildes);
	fd.fildes = -1;  // so it can't be closed again
	return nil
}

func (fd *FD) Read(b []byte) (ret int, err *os.Error) {
	if fd == nil {
		return -1, os.EINVAL
	}
	r, e := syscall.Read(fd.fildes, &b[0], int64(len(b)));
	return int(r), os.ErrnoToError(e)
}

func (fd *FD) Write(b []byte) (ret int, err *os.Error) {
	if fd == nil {
		return -1, os.EINVAL
	}
	r, e := syscall.Write(fd.fildes, &b[0], int64(len(b)));
	return int(r), os.ErrnoToError(e)
}

func (fd *FD) String() string {
	return fd.name
}

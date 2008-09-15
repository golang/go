// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fd

import Syscall "syscall"

export type FD struct {
	fildes	int64;	// file descriptor number
	name	string;	// file name at Open time
}

func NewFD(fd int64, name string) *FD {
	if fd < 0 {
		return nil
	}
	n := new(FD);
	n.fildes = fd;
	n.name = name;
	return n
}

export var (
	Stdin  = NewFD(0, "/dev/stdin");
	Stdout = NewFD(1, "/dev/stdout");
	Stderr = NewFD(2, "/dev/stderr");
)

export func Open(name string, mode int64, perm int64) (fd *FD, errno int64) {
	r, e := Syscall.open(name, mode, perm);
	return NewFD(r, name), e
}

func (fd *FD) Close() int64 {
	if fd == nil {
		return Syscall.EINVAL
	}
	r, e := Syscall.close(fd.fildes);
	fd.fildes = -1;  // so it can't be closed again
	return 0
}

func (fd *FD) Read(b *[]byte) (ret int64, errno int64) {
	if fd == nil {
		return -1, Syscall.EINVAL
	}
	r, e := Syscall.read(fd.fildes, &b[0], int64(len(b)));
	return r, e
}

func (fd *FD) Write(b *[]byte) (ret int64, errno int64) {
	if fd == nil {
		return -1, Syscall.EINVAL
	}
	r, e := Syscall.write(fd.fildes, &b[0], int64(len(b)));
	return r, e
}

func (fd *FD) Name() string {
	return fd.name
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package file

import (
	"os";
	"syscall";
)

type File struct {
	fd      int;  // file descriptor number
	name    string; // file name at Open time
}

func newFile(fd int, name string) *File {
	if fd < 0 {
		return nil
	}
	return &File{fd, name}
}

var (
	Stdin  = newFile(0, "/dev/stdin");
	Stdout = newFile(1, "/dev/stdout");
	Stderr = newFile(2, "/dev/stderr");
)

func Open(name string, mode int, perm int) (file *File, err os.Error) {
	r, e := syscall.Open(name, mode, perm);
	return newFile(r, name), os.ErrnoToError(e)
}

func (file *File) Close() os.Error {
	if file == nil {
		return os.EINVAL
	}
	e := syscall.Close(file.fd);
	file.fd = -1;  // so it can't be closed again
	return os.ErrnoToError(e);
}

func (file *File) Read(b []byte) (ret int, err os.Error) {
	if file == nil {
		return -1, os.EINVAL
	}
	r, e := syscall.Read(file.fd, b);
	return int(r), os.ErrnoToError(e)
}

func (file *File) Write(b []byte) (ret int, err os.Error) {
	if file == nil {
		return -1, os.EINVAL
	}
	r, e := syscall.Write(file.fd, b);
	return int(r), os.ErrnoToError(e)
}

func (file *File) String() string {
	return file.name
}

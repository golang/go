// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package file

import (
	"os"
	"syscall"
)

type File struct {
	fd   syscall.Handle // file descriptor number
	name string         // file name at Open time
}

func newFile(fd syscall.Handle, name string) *File {
	if fd == ^syscall.Handle(0) {
		return nil
	}
	return &File{fd, name}
}

var (
	Stdin  = newFile(syscall.Stdin, "/dev/stdin")
	Stdout = newFile(syscall.Stdout, "/dev/stdout")
	Stderr = newFile(syscall.Stderr, "/dev/stderr")
)

func OpenFile(name string, mode int, perm uint32) (file *File, err error) {
	r, err := syscall.Open(name, mode, perm)
	return newFile(r, name), err
}

const (
	O_RDONLY = syscall.O_RDONLY
	O_RDWR   = syscall.O_RDWR
	O_CREATE = syscall.O_CREAT
	O_TRUNC  = syscall.O_TRUNC
)

func Open(name string) (file *File, err error) {
	return OpenFile(name, O_RDONLY, 0)
}

func Create(name string) (file *File, err error) {
	return OpenFile(name, O_RDWR|O_CREATE|O_TRUNC, 0666)
}

func (file *File) Close() error {
	if file == nil {
		return os.EINVAL
	}
	err := syscall.Close(file.fd)
	file.fd = syscall.InvalidHandle // so it can't be closed again
	return err
}

func (file *File) Read(b []byte) (ret int, err error) {
	if file == nil {
		return -1, os.EINVAL
	}
	r, err := syscall.Read(file.fd, b)
	return int(r), err
}

func (file *File) Write(b []byte) (ret int, err error) {
	if file == nil {
		return -1, os.EINVAL
	}
	r, err := syscall.Write(file.fd, b)
	return int(r), err
}

func (file *File) String() string {
	return file.name
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import syscall "syscall"
import os "os"

// Auxiliary information if the FD describes a directory
type dirInfo struct {	// TODO(r): 6g bug means this can't be private
	buf	[]byte;	// buffer for directory I/O
	nbuf	int64;	// length of buf; return value from Getdirentries
	bufp	int64;	// location of next record in buf.
}

// FDs are wrappers for file descriptors
type FD struct {
	fd int64;
	name	string;
	dirinfo	*dirInfo;	// nil unless directory being read
}

func (fd *FD) Fd() int64 {
	return fd.fd
}

func (fd *FD) Name() string {
	return fd.name
}

func NewFD(fd int64, name string) *FD {
	if fd < 0 {
		return nil
	}
	return &FD{fd, name, nil}
}

var (
	Stdin = NewFD(0, "/dev/stdin");
	Stdout = NewFD(1, "/dev/stdout");
	Stderr = NewFD(2, "/dev/stderr");
)

const (
	O_RDONLY = syscall.O_RDONLY;
	O_WRONLY = syscall.O_WRONLY;
	O_RDWR = syscall.O_RDWR;
	O_APPEND = syscall.O_APPEND;
	O_ASYNC = syscall.O_ASYNC;
	O_CREAT = syscall.O_CREAT;
	O_NOCTTY = syscall.O_NOCTTY;
	O_NONBLOCK = syscall.O_NONBLOCK;
	O_NDELAY = O_NONBLOCK;
	O_SYNC = syscall.O_SYNC;
	O_TRUNC = syscall.O_TRUNC;
)

func Open(name string, mode int, flags int) (fd *FD, err *Error) {
	r, e := syscall.Open(name, int64(mode), int64(flags));
	return NewFD(r, name), ErrnoToError(e)
}

func (fd *FD) Close() *Error {
	if fd == nil {
		return EINVAL
	}
	r, e := syscall.Close(fd.fd);
	fd.fd = -1;  // so it can't be closed again
	return ErrnoToError(e)
}

func (fd *FD) Read(b []byte) (ret int, err *Error) {
	if fd == nil {
		return 0, EINVAL
	}
	var r, e int64;
	if len(b) > 0 {  // because we access b[0]
		r, e = syscall.Read(fd.fd, &b[0], int64(len(b)));
		if r < 0 {
			r = 0
		}
	}
	return int(r), ErrnoToError(e)
}

func (fd *FD) Write(b []byte) (ret int, err *Error) {
	if fd == nil {
		return 0, EINVAL
	}
	var r, e int64;
	if len(b) > 0 {  // because we access b[0]
		r, e = syscall.Write(fd.fd, &b[0], int64(len(b)));
		if r < 0 {
			r = 0
		}
	}
	return int(r), ErrnoToError(e)
}

func (fd *FD) Seek(offset int64, whence int) (ret int64, err *Error) {
	r, e := syscall.Seek(fd.fd, offset, int64(whence));
	if e != 0 {
		return -1, ErrnoToError(e)
	}
	if fd.dirinfo != nil && r != 0 {
		return -1, ErrnoToError(syscall.EISDIR)
	}
	return r, nil
}

func (fd *FD) WriteString(s string) (ret int, err *Error) {
	if fd == nil {
		return 0, EINVAL
	}
	r, e := syscall.Write(fd.fd, syscall.StringBytePtr(s), int64(len(s)));
	if r < 0 {
		r = 0
	}
	return int(r), ErrnoToError(e)
}

func Pipe() (fd1 *FD, fd2 *FD, err *Error) {
	var p [2]int64;
	r, e := syscall.Pipe(&p);
	if e != 0 {
		return nil, nil, ErrnoToError(e)
	}
	return NewFD(p[0], "|0"), NewFD(p[1], "|1"), nil
}

func Mkdir(name string, perm int) *Error {
	r, e := syscall.Mkdir(name, int64(perm));
	return ErrnoToError(e)
}

func Stat(name string) (dir *Dir, err *Error) {
	stat := new(syscall.Stat_t);
	r, e := syscall.Stat(name, stat);
	if e != 0 {
		return nil, ErrnoToError(e)
	}
	return dirFromStat(name, new(Dir), stat), nil
}

func Fstat(fd *FD) (dir *Dir, err *Error) {
	stat := new(syscall.Stat_t);
	r, e := syscall.Fstat(fd.fd, stat);
	if e != 0 {
		return nil, ErrnoToError(e)
	}
	return dirFromStat(fd.name, new(Dir), stat), nil
}

func Lstat(name string) (dir *Dir, err *Error) {
	stat := new(syscall.Stat_t);
	r, e := syscall.Lstat(name, stat);
	if e != 0 {
		return nil, ErrnoToError(e)
	}
	return dirFromStat(name, new(Dir), stat), nil
}

// Non-portable function defined in operating-system-dependent file.
func Readdirnames(fd *FD, count int) (names []string, err *os.Error)

// Negative count means read until EOF.
func Readdir(fd *FD, count int) (dirs []Dir, err *os.Error) {
	dirname := fd.name;
	if dirname == "" {
		dirname = ".";
	}
	dirname += "/";
	names, err1 := Readdirnames(fd, count);
	if err1 != nil {
		return nil, err1
	}
	dirs = make([]Dir, len(names));
	for i, filename := range names {
		dirp, err := Stat(dirname + filename);
		if dirp ==  nil || err != nil {
			dirs[i].Name = filename	// rest is already zeroed out
		} else {
			dirs[i] = *dirp
		}
	}
	return
}


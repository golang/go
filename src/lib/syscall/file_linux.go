// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

// File operations for Linux

import (
	"syscall";
	"unsafe";
)

const nameBufsize = 512

func Open(name string, mode int64, perm int64) (ret int64, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_OPEN, int64(uintptr(unsafe.Pointer(namebuf))), mode, perm);
	return r1, err;
}

func Creat(name string, perm int64) (ret int64, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_OPEN, int64(uintptr(unsafe.Pointer(namebuf))),  O_CREAT|O_WRONLY|O_TRUNC, perm);
	return r1, err;
}

func Close(fd int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_CLOSE, fd, 0, 0);
	return r1, err;
}

func Read(fd int64, buf *byte, nbytes int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_READ, fd, int64(uintptr(unsafe.Pointer(buf))), nbytes);
	return r1, err;
}

func Write(fd int64, buf *byte, nbytes int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_WRITE, fd, int64(uintptr(unsafe.Pointer(buf))), nbytes);
	return r1, err;
}

func Seek(fd int64, offset int64, whence int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_LSEEK, fd, offset, whence);
	return r1, err;
}

func Pipe(fds *[2]int64) (ret int64, errno int64) {
	var t [2] int32;
	r1, r2, err := Syscall(SYS_PIPE, int64(uintptr(unsafe.Pointer(&t[0]))), 0, 0);
	if r1 < 0 {
		return r1, err;
	}
	fds[0] = int64(t[0]);
	fds[1] = int64(t[1]);
	return 0, 0;
}

func Stat(name string, buf *Stat_t) (ret int64, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_STAT, int64(uintptr(unsafe.Pointer(namebuf))), int64(uintptr(unsafe.Pointer(buf))), 0);
	return r1, err;
}

func Lstat(name string, buf *Stat_t) (ret int64, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_LSTAT, int64(uintptr(unsafe.Pointer(namebuf))), int64(uintptr(unsafe.Pointer(buf))), 0);
	return r1, err;
}

func Fstat(fd int64, buf *Stat_t) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_FSTAT, fd, int64(uintptr(unsafe.Pointer(buf))), 0);
	return r1, err;
}

func Unlink(name string) (ret int64, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_UNLINK, int64(uintptr(unsafe.Pointer(namebuf))), 0, 0);
	return r1, err;
}

func Fcntl(fd, cmd, arg int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_FCNTL, fd, cmd, arg);
	return r1, err
}

func Mkdir(name string, perm int64) (ret int64, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_MKDIR, int64(uintptr(unsafe.Pointer(namebuf))), perm, 0);
	return r1, err;
}

func Dup2(fd1, fd2 int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_DUP2, fd1, fd2, 0);
	return r1, err;
}

func Getdents(fd int64, buf *Dirent, nbytes int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_GETDENTS64, fd, int64(uintptr(unsafe.Pointer(buf))), nbytes);
	return r1, err;
}

func Chdir(dir string) (ret int64, errno int64) {
	namebuf := StringBytePtr(dir);
	r1, r2, err := Syscall(SYS_CHDIR, int64(uintptr(unsafe.Pointer(namebuf))), 0, 0);
	return r1, err;
}

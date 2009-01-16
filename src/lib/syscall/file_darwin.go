// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// File operations for Darwin

package syscall

import (
	"syscall";
	"unsafe";
)

const nameBufsize = 512

export func Open(name string, mode int64, perm int64) (ret int64, errno int64) {
	var namebuf [nameBufsize]byte;
	if !StringToBytes(namebuf, name) {
		return -1, ENAMETOOLONG
	}
	r1, r2, err := Syscall(SYS_OPEN, int64(uintptr(unsafe.pointer(&namebuf[0]))), mode, perm);
	return r1, err;
}

export func Creat(name string, perm int64) (ret int64, errno int64) {
	var namebuf [nameBufsize]byte;
	if !StringToBytes(namebuf, name) {
		return -1, ENAMETOOLONG
	}
	r1, r2, err := Syscall(SYS_OPEN, int64(uintptr(unsafe.pointer(&namebuf[0]))), O_CREAT|O_WRONLY|O_TRUNC, perm);
	return r1, err;
}

export func Close(fd int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_CLOSE, fd, 0, 0);
	return r1, err;
}

export func Read(fd int64, buf *byte, nbytes int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_READ, fd, int64(uintptr(unsafe.pointer(buf))), nbytes);
	return r1, err;
}

export func Write(fd int64, buf *byte, nbytes int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_WRITE, fd, int64(uintptr(unsafe.pointer(buf))), nbytes);
	return r1, err;
}

export func Pipe(fds *[2]int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_PIPE, 0, 0, 0);
	if r1 < 0 {
		return r1, err;
	}
	fds[0] = r1;
	fds[1] = r2;
	return 0, 0;
}

export func Stat(name string, buf *Stat_t) (ret int64, errno int64) {
	var namebuf [nameBufsize]byte;
	if !StringToBytes(namebuf, name) {
		return -1, ENAMETOOLONG
	}
	r1, r2, err := Syscall(SYS_STAT64, int64(uintptr(unsafe.pointer(&namebuf[0]))), int64(uintptr(unsafe.pointer(buf))), 0);
	return r1, err;
}

export func Lstat(name *byte, buf *Stat_t) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_LSTAT, int64(uintptr(unsafe.pointer(name))), int64(uintptr(unsafe.pointer(buf))), 0);
	return r1, err;
}

export func Fstat(fd int64, buf *Stat_t) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_FSTAT, fd, int64(uintptr(unsafe.pointer(buf))), 0);
	return r1, err;
}

export func Unlink(name string) (ret int64, errno int64) {
	var namebuf [nameBufsize]byte;
	if !StringToBytes(namebuf, name) {
		return -1, ENAMETOOLONG
	}
	r1, r2, err := Syscall(SYS_UNLINK, int64(uintptr(unsafe.pointer(&namebuf[0]))), 0, 0);
	return r1, err;
}

export func Fcntl(fd, cmd, arg int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_FCNTL, fd, cmd, arg);
	return r1, err
}

export func Mkdir(name string, perm int64) (ret int64, errno int64) {
	var namebuf [nameBufsize]byte;
	if !StringToBytes(namebuf, name) {
		return -1, ENAMETOOLONG
	}
	r1, r2, err := Syscall(SYS_MKDIR, int64(uintptr(unsafe.pointer(&namebuf[0]))), perm, 0);
	return r1, err;
}

export func Dup2(fd1, fd2 int64) (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_DUP2, fd1, fd2, 0);
	return r1, err;
}


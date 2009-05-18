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

func Open(name string, mode int64, perm int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_OPEN, int64(uintptr(unsafe.Pointer(namebuf))), mode, perm);
	return r1, err;
}

func Creat(name string, perm int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_OPEN, int64(uintptr(unsafe.Pointer(namebuf))), O_CREAT|O_WRONLY|O_TRUNC, perm);
	return r1, err;
}

func Close(fd int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_CLOSE, fd, 0, 0);
	return r1, err;
}

func Read(fd int64, buf *byte, nbytes int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_READ, fd, int64(uintptr(unsafe.Pointer(buf))), nbytes);
	return r1, err;
}

func Write(fd int64, buf *byte, nbytes int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_WRITE, fd, int64(uintptr(unsafe.Pointer(buf))), nbytes);
	return r1, err;
}

func Seek(fd int64, offset int64, whence int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_LSEEK, fd, offset, whence);
	return r1, err;
}

func Pipe(fds *[2]int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_PIPE, 0, 0, 0);
	if r1 < 0 {
		return r1, err;
	}
	fds[0] = r1;
	fds[1] = r2;
	return 0, 0;
}

func Stat(name string, buf *Stat_t) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_STAT64, int64(uintptr(unsafe.Pointer(namebuf))), int64(uintptr(unsafe.Pointer(buf))), 0);
	return r1, err;
}

func Lstat(name string, buf *Stat_t) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_LSTAT64, int64(uintptr(unsafe.Pointer(namebuf))), int64(uintptr(unsafe.Pointer(buf))), 0);
	return r1, err;
}

func Fstat(fd int64, buf *Stat_t) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_FSTAT64, fd, int64(uintptr(unsafe.Pointer(buf))), 0);
	return r1, err;
}

func Unlink(name string) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_UNLINK, int64(uintptr(unsafe.Pointer(namebuf))), 0, 0);
	return r1, err;
}

func Rmdir(name string) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_RMDIR, int64(uintptr(unsafe.Pointer(namebuf))), 0, 0);
	return r1, err;
}

func Fcntl(fd, cmd, arg int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_FCNTL, fd, cmd, arg);
	return r1, err
}

func Mkdir(name string, perm int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_MKDIR, int64(uintptr(unsafe.Pointer(namebuf))), perm, 0);
	return r1, err;
}

func Dup2(fd1, fd2 int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_DUP2, fd1, fd2, 0);
	return r1, err;
}

func Getdirentries(fd int64, buf *byte, nbytes int64, basep *int64) (ret, errno int64) {
	r1, r2, err := Syscall6(SYS_GETDIRENTRIES64, fd, int64(uintptr(unsafe.Pointer(buf))), nbytes, int64(uintptr(unsafe.Pointer(basep))), 0, 0);
	return r1, err;
}

func Chdir(dir string) (ret, errno int64) {
	namebuf := StringBytePtr(dir);
	r1, r2, err := Syscall(SYS_CHDIR, int64(uintptr(unsafe.Pointer(namebuf))), 0, 0);
	return r1, err;
}

func Fchdir(fd int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_FCHDIR, fd, 0, 0);
	return r1, err;
}

func Link(oldname, newname string) (ret, errno int64) {
	oldbuf := StringBytePtr(oldname);
	newbuf := StringBytePtr(newname);
	r1, r2, err := Syscall(SYS_LINK, int64(uintptr(unsafe.Pointer(oldbuf))), int64(uintptr(unsafe.Pointer(newbuf))), 0);
	return r1, err;
}

func Symlink(oldname, newname string) (ret, errno int64) {
	oldbuf := StringBytePtr(oldname);
	newbuf := StringBytePtr(newname);
	r1, r2, err := Syscall(SYS_SYMLINK, int64(uintptr(unsafe.Pointer(oldbuf))), int64(uintptr(unsafe.Pointer(newbuf))), 0);
	return r1, err;
}

func Readlink(name string, buf *byte, nbytes int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_READLINK, int64(uintptr(unsafe.Pointer(namebuf))), int64(uintptr(unsafe.Pointer(buf))), nbytes);
	return r1, err;
}

func Chmod(name string, mode int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_CHMOD, int64(uintptr(unsafe.Pointer(namebuf))), mode, 0);
	return r1, err;
}

func Fchmod(fd, mode int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_FCHMOD, fd, mode, 0);
	return r1, err;
}

func Chown(name string, uid, gid int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_CHOWN, int64(uintptr(unsafe.Pointer(namebuf))), uid, gid);
	return r1, err;
}

func Lchown(name string, uid, gid int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_LCHOWN, int64(uintptr(unsafe.Pointer(namebuf))), uid, gid);
	return r1, err;
}

func Fchown(fd, uid, gid int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_FCHOWN, fd, uid, gid);
	return r1, err;
}

func Truncate(name string, length int64) (ret, errno int64) {
	namebuf := StringBytePtr(name);
	r1, r2, err := Syscall(SYS_TRUNCATE, int64(uintptr(unsafe.Pointer(namebuf))), length, 0);
	return r1, err;
}

func Ftruncate(fd, length int64) (ret, errno int64) {
	r1, r2, err := Syscall(SYS_FTRUNCATE, fd, length, 0);
	return r1, err;
}

// The const provides a compile-time constant so clients
// can adjust to whether there is a working Getwd and avoid
// even linking this function into the binary.  See ../os/getwd.go.
const ImplementsGetwd = false

func Getwd() (string, int64) {
	return "", ENOTSUP;
}


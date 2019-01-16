// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "unsafe"

func setTimespec(sec, nsec int64) Timespec {
	return Timespec{Sec: int64(sec), Nsec: int64(nsec)}
}

func setTimeval(sec, usec int64) Timeval {
	return Timeval{Sec: int64(sec), Usec: int32(usec)}
}

//sys	closedir(dir uintptr) (err error)
//sys	Fstat(fd int, stat *Stat_t) (err error)
//sys	Fstatfs(fd int, stat *Statfs_t) (err error)
//sysnb	Gettimeofday(tp *Timeval) (err error)
//sys	Lstat(path string, stat *Stat_t) (err error)
//sys	readdir_r(dirp uintptr, entry uintptr, result uintptr) (res int)
//sys	Stat(path string, stat *Stat_t) (err error)
//sys	Statfs(path string, stat *Statfs_t) (err error)
//sys   fstatat(fd int, path string, stat *Stat_t, flags int) (err error)

func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	return 0, ENOSYS
}

func SetKevent(k *Kevent_t, fd, mode, flags int) {
	k.Ident = uint64(fd)
	k.Filter = int16(mode)
	k.Flags = uint16(flags)
}

func (iov *Iovec) SetLen(length int) {
	iov.Len = uint64(length)
}

func (msghdr *Msghdr) SetControllen(length int) {
	msghdr.Controllen = uint32(length)
}

func (cmsg *Cmsghdr) SetLen(length int) {
	cmsg.Len = uint32(length)
}

func sendfile(outfd int, infd int, offset *int64, count int) (written int, err error) {
	var length = uint64(count)

	_, _, e1 := syscall6(funcPC(libc_sendfile_trampoline), uintptr(infd), uintptr(outfd), uintptr(*offset), uintptr(unsafe.Pointer(&length)), 0, 0)

	written = int(length)

	if e1 != 0 {
		err = e1
	}
	return
}

func libc_sendfile_trampoline()

//go:linkname libc_sendfile libc_sendfile
//go:cgo_import_dynamic libc_sendfile sendfile "/usr/lib/libSystem.B.dylib"

func fdopendir(fd int) (dir uintptr, err error) {
	r0, _, e1 := syscallXPtr(funcPC(libc_fdopendir_trampoline), uintptr(fd), 0, 0)
	dir = uintptr(r0)
	if e1 != 0 {
		err = errnoErr(e1)
	}
	return
}

func libc_fdopendir_trampoline()

//go:linkname libc_fdopendir libc_fdopendir
//go:cgo_import_dynamic libc_fdopendir fdopendir "/usr/lib/libSystem.B.dylib"

// Implemented in the runtime package (runtime/sys_darwin_64.go)
func syscallX(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)
func syscallXPtr(fn, a1, a2, a3 uintptr) (r1, r2 uintptr, err Errno)

func Syscall9(num, a1, a2, a3, a4, a5, a6, a7, a8, a9 uintptr) (r1, r2 uintptr, err Errno) // sic

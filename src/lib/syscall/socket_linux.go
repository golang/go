// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Low-level socket interface.
// Only for implementing net package.
// DO NOT USE DIRECTLY.

package syscall
import (
	"syscall";
	"unsafe";
)

func SockaddrToSockaddrInet4(s *Sockaddr) *SockaddrInet4;
func SockaddrToSockaddrInet6(s *Sockaddr) *SockaddrInet6;
func SockaddrInet4ToSockaddr(s *SockaddrInet4) *Sockaddr;
func SockaddrInet6ToSockaddr(s *SockaddrInet6) *Sockaddr;

func saLen(s *Sockaddr) int64 {
	switch s.Family {
	case AF_UNIX:
		return SizeofSockaddrUnix;
	case AF_INET:
		return SizeofSockaddrInet4;
	case AF_INET6:
		return SizeofSockaddrInet6
	}
	return 0
}

func Socket(domain, proto, typ int64) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_SOCKET, domain, proto, typ);
	return r1, e
}

func Connect(fd int64, sa *Sockaddr) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_CONNECT, fd, int64(uintptr(unsafe.Pointer(sa))), saLen(sa));
	return r1, e
}

func Bind(fd int64, sa *Sockaddr) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_BIND, fd, int64(uintptr(unsafe.Pointer(sa))), saLen(sa));
	return r1, e
}

func Listen(fd, n int64) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_LISTEN, fd, n, 0);
	return r1, e
}

func Accept(fd int64, sa *Sockaddr) (ret int64, err int64) {
	var n int32 = SizeofSockaddr;
	r1, r2, e := Syscall(SYS_ACCEPT, fd, int64(uintptr(unsafe.Pointer(sa))), int64(uintptr(unsafe.Pointer(&n))));
	return r1, e
}

func Setsockopt(fd, level, opt, valueptr, length int64) (ret int64, err int64) {
	if fd < 0 {
		return -1, EINVAL
	}
	r1, r2, e := Syscall6(SYS_SETSOCKOPT, fd, level, opt, valueptr, length, 0);
	return r1, e
}

func Setsockopt_int(fd, level, opt int64, value int) int64 {
	n := int32(opt);
	r1, e := Setsockopt(fd, level, opt, int64(uintptr(unsafe.Pointer(&n))), 4);
	return e
}

func Setsockopt_tv(fd, level, opt, nsec int64) int64 {
	var tv Timeval;
	nsec += 999;
	tv.Sec = int64(nsec/1000000000);
	tv.Usec = uint64(nsec%1000000000);
	r1, e := Setsockopt(fd, level, opt, int64(uintptr(unsafe.Pointer(&tv))), 4);
	return e
}

func Setsockopt_linger(fd, level, opt int64, sec int) int64 {
	var l Linger;
	if sec != 0 {
		l.Yes = 1;
		l.Sec = int32(sec)
	} else {
		l.Yes = 0;
		l.Sec = 0
	}
	r1, err := Setsockopt(fd, level, opt, int64(uintptr(unsafe.Pointer(&l))), 8);
	return err
}

/*
func getsockopt(fd, level, opt, valueptr, lenptr int64) (ret int64, errno int64) {
	r1, r2, err := Syscall6(GETSOCKOPT, fd, level, opt, valueptr, lenptr, 0);
	return r1, err;
}
*/

func Epoll_create(size int64) (ret int64, errno int64) {
	r1, r2, err := syscall.Syscall(SYS_EPOLL_CREATE, size, 0, 0);
	return r1, err
}

func Epoll_ctl(epfd, op, fd int64, ev *EpollEvent) int64 {
	r1, r2, err := syscall.Syscall6(SYS_EPOLL_CTL, epfd, op, fd, int64(uintptr(unsafe.Pointer(ev))), 0, 0);
	return err
}

func Epoll_wait(epfd int64, ev []EpollEvent, msec int64) (ret int64, err int64) {
	var evptr, nev int64;
	if ev != nil && len(ev) > 0 {
		nev = int64(len(ev));
		evptr = int64(uintptr(unsafe.Pointer(&ev[0])))
	}
	r1, r2, err1 := syscall.Syscall6(SYS_EPOLL_WAIT, epfd, evptr, nev, msec, 0, 0);
	return r1, err1
}


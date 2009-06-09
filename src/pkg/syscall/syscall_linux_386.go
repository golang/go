// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import (
	"syscall";
	"unsafe";
)

func Getpagesize() int {
	return 4096
}

func TimespecToNsec(ts Timespec) int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec);
}

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = int32(nsec / 1e9);
	ts.Nsec = int32(nsec % 1e9);
	return;
}

func TimevalToNsec(tv Timeval) int64 {
	return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3;
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999;	// round up to microsecond
	tv.Sec = int32(nsec/1e9);
	tv.Usec = int32(nsec%1e9 / 1e3);
	return;
}

// On x86 Linux, all the socket calls go through an extra indirection,
// I think because the 5-register system call interface can't handle
// the 6-argument calls like sendto and recvfrom.  Instead the
// arguments to the underlying system call are the number below
// and a pointer to an array of uintptr.  We hide the pointer in the
// socketcall assembly to avoid allocation on every system call.

const (
	// see linux/net.h
	_SOCKET = 1;
	_BIND = 2;
	_CONNECT = 3;
	_LISTEN = 4;
	_ACCEPT = 5;
	_GETSOCKNAME = 6;
	_GETPEERNAME = 7;
	_SOCKETPAIR = 8;
	_SEND = 9;
	_RECV = 10;
	_SENDTO = 11;
	_RECVFROM = 12;
	_SHUTDOWN = 13;
	_SETSOCKOPT = 14;
	_GETSOCKOPT = 15;
	_SENDMSG = 16;
	_RECVMSG = 17;
)

func socketcall(call int, a0, a1, a2, a3, a4, a5 uintptr) (n int, errno int)

func accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int) {
	fd, errno = socketcall(_SOCKET, uintptr(s), uintptr(unsafe.Pointer(rsa)), uintptr(unsafe.Pointer(addrlen)), 0, 0, 0);
	return;
}

func bind(s int, addr uintptr, addrlen _Socklen) (errno int) {
	var _ int;
	_, errno = socketcall(_BIND, uintptr(s), uintptr(addr), uintptr(addrlen), 0, 0, 0);
	return;
}

func connect(s int, addr uintptr, addrlen _Socklen) (errno int) {
	var _ int;
	_, errno = socketcall(_CONNECT, uintptr(s), uintptr(addr), uintptr(addrlen), 0, 0, 0);
	return;
}

func socket(domain int, typ int, proto int) (fd int, errno int) {
	fd, errno = socketcall(_SOCKET, uintptr(domain), uintptr(typ), uintptr(proto), 0, 0, 0);
	return;
}

func setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int) {
	var _ int;
	_, errno = socketcall(_SETSOCKOPT, uintptr(s), uintptr(level), uintptr(name), uintptr(val), uintptr(vallen), 0);
	return;
}

func Listen(s int, n int) (errno int) {
	var _ int;
	_, errno = socketcall(_LISTEN, uintptr(s), uintptr(n), 0, 0, 0, 0);
	return;
}


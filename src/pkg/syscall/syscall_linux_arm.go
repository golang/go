// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = int32(nsec / 1e9);
	ts.Nsec = int32(nsec % 1e9);
	return;
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999;	// round up to microsecond
	tv.Sec = int32(nsec/1e9);
	tv.Usec = int32(nsec%1e9 / 1e3);
	return;
}

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int)
//sys	bind(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	connect(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	getgroups(n int, list *_Gid_t) (nn int, errno int) = SYS_GETGROUPS32
//sys	setgroups(n int, list *_Gid_t) (errno int) = SYS_SETGROUPS32
//sys	setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int)
//sys	socket(domain int, typ int, proto int) (fd int, errno int)
//sys	getpeername(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)
//sys	getsockname(fd int, rsa *RawSockaddrAny, addrlen *_Socklen) (errno int)

//sys	Select(nfd int, r *FdSet, w *FdSet, e *FdSet, timeout *Timeval) (n int, errno int) = SYS__NEWSELECT


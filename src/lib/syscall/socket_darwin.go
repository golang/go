// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Low-level socket interface.
// Only for implementing net package.
// DO NOT USE DIRECTLY.

package syscall
import "syscall"

export func SockaddrToSockaddrInet4(s *Sockaddr) *SockaddrInet4;
export func SockaddrToSockaddrInet6(s *Sockaddr) *SockaddrInet6;
export func SockaddrInet4ToSockaddr(s *SockaddrInet4) *Sockaddr;
export func SockaddrInet6ToSockaddr(s *SockaddrInet6) *Sockaddr;

export func socket(domain, proto, typ int64) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_SOCKET, domain, proto, typ);
	return r1, e
}

export func connect(fd int64, sa *Sockaddr) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_CONNECT, fd, SockaddrPtr(sa), int64(sa.len));
	return r1, e
}

export func bind(fd int64, sa *Sockaddr) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_BIND, fd, SockaddrPtr(sa), int64(sa.len));
	return r1, e
}

export func listen(fd, n int64) (ret int64, err int64) {
	r1, r2, e := Syscall(SYS_LISTEN, fd, n, 0);
	return r1, e
}

export func accept(fd int64, sa *Sockaddr) (ret int64, err int64) {
	n := SizeofSockaddr;
	r1, r2, e := Syscall(SYS_ACCEPT, fd, SockaddrPtr(sa), Int32Ptr(&n));
	return r1, e
}

export func setsockopt(fd, level, opt, valueptr, length int64) (ret int64, err int64) {
	if fd < 0 {
		return -1, EINVAL
	}
	r1, r2, e := Syscall6(SYS_SETSOCKOPT, fd, level, opt, valueptr, length, 0);
	return r1, e
}

export func setsockopt_int(fd, level, opt int64, value int) int64 {
	n := int(opt);
	r1, e := setsockopt(fd, level, opt, Int32Ptr(&n), 4);
	return e
}

export func setsockopt_tv(fd, level, opt, nsec int64) int64 {
	var tv Timeval;
	nsec += 999;
	tv.sec = int64(nsec/1000000000);
	tv.usec = uint(nsec%1000000000);
	r1, e := setsockopt(fd, level, opt, TimevalPtr(&tv), 4);
	return e
}

export func setsockopt_linger(fd, level, opt int64, sec int) int64 {
	var l Linger;
	if sec != 0 {
		l.yes = 1;
		l.sec = sec
	} else {
		l.yes = 0;
		l.sec = 0
	}
	r1, err := setsockopt(fd, level, opt, LingerPtr(&l), 8);
	return err
}

/*
export func getsockopt(fd, level, opt, valueptr, lenptr int64) (ret int64, errno int64) {
	r1, r2, err := Syscall6(SYS_GETSOCKOPT, fd, level, opt, valueptr, lenptr, 0);
	return r1, err;
}
*/

export func kqueue() (ret int64, errno int64) {
	r1, r2, err := Syscall(SYS_KQUEUE, 0, 0, 0);
	return r1, err
}

export func kevent(kq int64, changes, events *[]Kevent, timeout *Timespec) (ret int64, errno int64) {
	var nchange, changeptr, nevent, eventptr int64;
	nchange = 0;
	changeptr = 0;
	nevent = 0;
	eventptr = 0;
	if changes != nil && len(changes) > 0 {
		changeptr = KeventPtr(&changes[0]);
		nchange = int64(len(changes))
	}
	if events != nil && len(events) > 0 {
		eventptr = KeventPtr(&events[0]);
		nevent = int64(len(events))
	}
	r1, r2, err := Syscall6(SYS_KEVENT, kq, changeptr, nchange,
		eventptr, nevent, TimespecPtr(timeout));
	return r1, err
}


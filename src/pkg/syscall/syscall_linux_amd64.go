// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syscall

import "syscall"

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, errno int)
//sys	bind(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	connect(s int, addr uintptr, addrlen _Socklen) (errno int)
//sys	socket(domain int, typ int, proto int) (fd int, errno int)
//sys	setsockopt(s int, level int, name int, val uintptr, vallen int) (errno int)
//sys	Listen(s int, n int) (errno int)
//sys	Shutdown(fd int, how int) (errno int)

func Getpagesize() int {
	return 4096
}

func TimespecToNsec(ts Timespec) int64 {
	return int64(ts.Sec)*1e9 + int64(ts.Nsec);
}

func NsecToTimespec(nsec int64) (ts Timespec) {
	ts.Sec = nsec / 1e9;
	ts.Nsec = nsec % 1e9;
	return;
}

func TimevalToNsec(tv Timeval) int64 {
	return int64(tv.Sec)*1e9 + int64(tv.Usec)*1e3;
}

func NsecToTimeval(nsec int64) (tv Timeval) {
	nsec += 999;	// round up to microsecond
	tv.Sec = nsec/1e9;
	tv.Usec = nsec%1e9 / 1e3;
	return;
}


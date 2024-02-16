// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We require Linux kernel version 2.6.32. The accept4 system call was
// added in version 2.6.28, so in general we can use accept4.
// Unfortunately, for ARM only, accept4 was added in version 2.6.36.
// Handle that case here, by using a copy of the Accept function that
// we used in Go 1.17.

//go:build linux && arm

package syscall

//sys	accept(s int, rsa *RawSockaddrAny, addrlen *_Socklen) (fd int, err error)

func Accept(fd int) (nfd int, sa Sockaddr, err error) {
	var rsa RawSockaddrAny
	var len _Socklen = SizeofSockaddrAny
	// Try accept4 first for Android and newer kernels.
	nfd, err = accept4(fd, &rsa, &len, 0)
	if err == ENOSYS {
		nfd, err = accept(fd, &rsa, &len)
	}
	if err != nil {
		return
	}
	sa, err = anyToSockaddr(&rsa)
	if err != nil {
		Close(nfd)
		nfd = 0
	}
	return
}

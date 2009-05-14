// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"net";
	"os";
	"syscall";
	"unsafe";
)

func v4ToSockaddr(p IP, port int) (sa1 *syscall.Sockaddr, err os.Error) {
	p = p.To4();
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}
	sa := new(syscall.SockaddrInet4);
	sa.Family = syscall.AF_INET;
	sa.Port[0] = byte(port>>8);
	sa.Port[1] = byte(port);
	for i := 0; i < IPv4len; i++ {
		sa.Addr[i] = p[i]
	}
	return (*syscall.Sockaddr)(unsafe.Pointer(sa)), nil
}

func v6ToSockaddr(p IP, port int) (sa1 *syscall.Sockaddr, err os.Error) {
	p = p.To16();
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}

	// IPv4 callers use 0.0.0.0 to mean "announce on any available address".
	// In IPv6 mode, Linux treats that as meaning "announce on 0.0.0.0",
	// which it refuses to do.  Rewrite to the IPv6 all zeros.
	if p4 := p.To4(); p4 != nil && p4[0] == 0 && p4[1] == 0 && p4[2] == 0 && p4[3] == 0 {
		p = IPzero;
	}

	sa := new(syscall.SockaddrInet6);
	sa.Family = syscall.AF_INET6;
	sa.Port[0] = byte(port>>8);
	sa.Port[1] = byte(port);
	for i := 0; i < IPv6len; i++ {
		sa.Addr[i] = p[i]
	}
	return (*syscall.Sockaddr)(unsafe.Pointer(sa)), nil
}

func sockaddrToIP(sa1 *syscall.Sockaddr) (p IP, port int, err os.Error) {
	switch sa1.Family {
	case syscall.AF_INET:
		sa := (*syscall.SockaddrInet4)(unsafe.Pointer(sa1));
		a := IP(&sa.Addr).To16();
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return a, int(sa.Port[0])<<8 + int(sa.Port[1]), nil;
	case syscall.AF_INET6:
		sa := (*syscall.SockaddrInet6)(unsafe.Pointer(sa1));
		a := IP(&sa.Addr).To16();
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return a, int(sa.Port[0])<<8 + int(sa.Port[1]), nil;
	default:
		return nil, 0, os.EINVAL
	}
	return nil, 0, nil	// not reached
}

func listenBacklog() int64 {
	// TODO: Read the limit from /proc/sys/net/core/somaxconn,
	// to take advantage of kernels that have raised the limit.
	return syscall.SOMAXCONN
}

func unixToSockaddr(name string) (sa1 *syscall.Sockaddr, err os.Error) {
	sa := new(syscall.SockaddrUnix);
	n := len(name);
	if n >= len(sa.Path) || n == 0 {
		return nil, os.EINVAL;
	}
	sa.Family = syscall.AF_UNIX;
	for i := 0; i < len(name); i++ {
		sa.Path[i] = name[i];
	}

	// Special case: @ in first position indicates
	// an abstract socket, which has no file system
	// representation and starts with a NUL byte
	// when talking to the kernel about it.
	if sa.Path[0] == '@' {
		sa.Path[0] = 0;
	}
	sa.Length = 1 + int64(n) + 1;	// family, name, \0

	return (*syscall.Sockaddr)(unsafe.Pointer(sa)), nil;
}

func sockaddrToUnix(sa1 *syscall.Sockaddr) (string, os.Error) {
	if sa1.Family != syscall.AF_UNIX {
		return "", os.EINVAL;
	}

	sa := (*syscall.SockaddrUnix)(unsafe.Pointer(sa1));

	// @ special case (see comment in unixToSockaddr).
	if sa.Path[0] == 0 {
		// Not friendly to overwrite in place but
		// okay in an internal function.
		// The caller doesn't care if we do.
		sa.Path[0] = '@';
	}

	// count length of path
	n := 0;
	for n < len(sa.Path) && sa.Path[n] != 0 {
		n++;
	}
	return string(sa.Path[0:n]), nil;
}

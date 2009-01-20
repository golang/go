// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os";
	"syscall";
	"net";
	"unsafe";
)

func IPv4ToSockaddr(p []byte, port int) (sa1 *syscall.Sockaddr, err *os.Error) {
	p = ToIPv4(p);
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}
	sa := new(syscall.SockaddrInet4);
	sa.Len = syscall.SizeofSockaddrInet4;
	sa.Family = syscall.AF_INET;
	sa.Port[0] = byte(port>>8);
	sa.Port[1] = byte(port);
	for i := 0; i < IPv4len; i++ {
		sa.Addr[i] = p[i]
	}
	return unsafe.Pointer(sa).(*syscall.Sockaddr), nil
}

func IPv6ToSockaddr(p []byte, port int) (sa1 *syscall.Sockaddr, err *os.Error) {
	p = ToIPv6(p);
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}
	sa := new(syscall.SockaddrInet6);
	sa.Len = syscall.SizeofSockaddrInet6;
	sa.Family = syscall.AF_INET6;
	sa.Port[0] = byte(port>>8);
	sa.Port[1] = byte(port);
	for i := 0; i < IPv6len; i++ {
		sa.Addr[i] = p[i]
	}
	return unsafe.Pointer(sa).(*syscall.Sockaddr), nil
}


func SockaddrToIP(sa1 *syscall.Sockaddr) (p []byte, port int, err *os.Error) {
	switch sa1.Family {
	case syscall.AF_INET:
		sa := unsafe.Pointer(sa1).(*syscall.SockaddrInet4);
		a := ToIPv6(sa.Addr);
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return a, int(sa.Port[0])<<8 + int(sa.Port[1]), nil;
	case syscall.AF_INET6:
		sa := unsafe.Pointer(sa1).(*syscall.SockaddrInet6);
		a := ToIPv6(sa.Addr);
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return nil, int(sa.Port[0])<<8 + int(sa.Port[1]), nil;
	default:
		return nil, 0, os.EINVAL
	}
	return nil, 0, nil	// not reached
}

func ListenBacklog() int64 {
	return syscall.SOMAXCONN
}


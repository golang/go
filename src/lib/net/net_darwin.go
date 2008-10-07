// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os";
	"syscall";
	"net"
)

export func IPv4ToSockaddr(p *[]byte, port int) (sa1 *syscall.Sockaddr, err *os.Error) {
	p = ToIPv4(p);
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}
	sa := new(syscall.SockaddrInet4);
	sa.len = syscall.SizeofSockaddrInet4;
	sa.family = syscall.AF_INET;
	sa.port[0] = byte(port>>8);
	sa.port[1] = byte(port);
	for i := 0; i < IPv4len; i++ {
		sa.addr[i] = p[i]
	}
	return syscall.SockaddrInet4ToSockaddr(sa), nil
}

export func IPv6ToSockaddr(p *[]byte, port int) (sa1 *syscall.Sockaddr, err *os.Error) {
	p = ToIPv6(p);
	if p == nil || port < 0 || port > 0xFFFF {
		return nil, os.EINVAL
	}
	sa := new(syscall.SockaddrInet6);
	sa.len = syscall.SizeofSockaddrInet6;
	sa.family = syscall.AF_INET6;
	sa.port[0] = byte(port>>8);
	sa.port[1] = byte(port);
	for i := 0; i < IPv6len; i++ {
		sa.addr[i] = p[i]
	}
	return syscall.SockaddrInet6ToSockaddr(sa), nil
}

export func SockaddrToIP(sa1 *syscall.Sockaddr) (p *[]byte, port int, err *os.Error) {
	switch sa1.family {
	case syscall.AF_INET:
		sa := syscall.SockaddrToSockaddrInet4(sa1);
		a := ToIPv6(&sa.addr);
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return a, int(sa.port[0])<<8 + int(sa.port[1]), nil;
	case syscall.AF_INET6:
		sa := syscall.SockaddrToSockaddrInet6(sa1);
		a := ToIPv6(&sa.addr);
		if a == nil {
			return nil, 0, os.EINVAL
		}
		return a, int(sa.port[0])<<8 + int(sa.port[1]), nil;
	default:
		return nil, 0, os.EINVAL
	}
	return nil, 0, nil	// not reached
}

export func ListenBacklog() int64 {
	return syscall.SOMAXCONN
}


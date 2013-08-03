// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package net

import "syscall"

func listenerSockaddr(s, f int, laddr sockaddr) (syscall.Sockaddr, error) {
	switch laddr := laddr.(type) {
	case *TCPAddr, *UnixAddr:
		if err := setDefaultListenerSockopts(s); err != nil {
			return nil, err
		}
		return laddr.sockaddr(f)
	case *UDPAddr:
		if laddr.IP != nil && laddr.IP.IsMulticast() {
			if err := setDefaultMulticastSockopts(s); err != nil {
				return nil, err
			}
			addr := *laddr
			switch f {
			case syscall.AF_INET:
				addr.IP = IPv4zero
			case syscall.AF_INET6:
				addr.IP = IPv6unspecified
			}
			laddr = &addr
		}
		return laddr.sockaddr(f)
	default:
		return laddr.sockaddr(f)
	}
}

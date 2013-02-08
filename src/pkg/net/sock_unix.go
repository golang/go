// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package net

import "syscall"

func listenerSockaddr(s, f int, la syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (syscall.Sockaddr, error) {
	a := toAddr(la)
	if a == nil {
		return la, nil
	}
	switch a := a.(type) {
	case *TCPAddr, *UnixAddr:
		if err := setDefaultListenerSockopts(s); err != nil {
			return nil, err
		}
	case *UDPAddr:
		if a.IP.IsMulticast() {
			if err := setDefaultMulticastSockopts(s); err != nil {
				return nil, err
			}
			switch f {
			case syscall.AF_INET:
				a.IP = IPv4zero
			case syscall.AF_INET6:
				a.IP = IPv6unspecified
			}
			return a.sockaddr(f)
		}
	}
	return la, nil
}

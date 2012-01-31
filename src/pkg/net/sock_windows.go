// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Sockets for Windows

package net

import "syscall"

func maxListenerBacklog() int {
	// TODO: Implement this
	return syscall.SOMAXCONN
}

func listenerSockaddr(s syscall.Handle, f int, la syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (syscall.Sockaddr, error) {
	a := toAddr(la)
	if a == nil {
		return la, nil
	}
	switch v := a.(type) {
	case *UDPAddr:
		if v.IP.IsMulticast() {
			err := setDefaultMulticastSockopts(s)
			if err != nil {
				return nil, err
			}
			switch f {
			case syscall.AF_INET:
				v.IP = IPv4zero
			case syscall.AF_INET6:
				v.IP = IPv6unspecified
			}
			return v.sockaddr(f)
		}
	}
	return la, nil
}

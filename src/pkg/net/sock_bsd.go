// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

// Sockets for BSD variants

package net

import (
	"runtime"
	"syscall"
)

func maxListenerBacklog() int {
	var (
		n   uint32
		err error
	)
	switch runtime.GOOS {
	case "darwin", "freebsd":
		n, err = syscall.SysctlUint32("kern.ipc.somaxconn")
	case "netbsd":
		// NOTE: NetBSD has no somaxconn-like kernel state so far
	case "openbsd":
		n, err = syscall.SysctlUint32("kern.somaxconn")
	}
	if n == 0 || err != nil {
		return syscall.SOMAXCONN
	}
	return int(n)
}

func listenerSockaddr(s, f int, la syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (syscall.Sockaddr, error) {
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

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "syscall"

func maxListenerBacklog() int {
	// TODO: Implement this
	// NOTE: Never return a number bigger than 1<<16 - 1. See issue 5030.
	return syscall.SOMAXCONN
}

func listenerSockaddr(s syscall.Handle, f int, la syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (syscall.Sockaddr, error) {
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

func sysSocket(f, t, p int) (syscall.Handle, error) {
	// See ../syscall/exec_unix.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err := syscall.Socket(f, t, p)
	if err == nil {
		syscall.CloseOnExec(s)
	}
	syscall.ForkLock.RUnlock()
	return s, err
}

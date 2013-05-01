// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

package net

import (
	"syscall"
	"time"
)

var listenerBacklog = maxListenerBacklog()

// Generic POSIX socket creation.
func socket(net string, f, t, p int, ipv6only bool, ulsa, ursa syscall.Sockaddr, deadline time.Time, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err error) {
	s, err := sysSocket(f, t, p)
	if err != nil {
		return nil, err
	}

	if err = setDefaultSockopts(s, f, t, ipv6only); err != nil {
		closesocket(s)
		return nil, err
	}

	// This socket is used by a listener.
	if ulsa != nil && ursa == nil {
		// We provide a socket that listens to a wildcard
		// address with reusable UDP port when the given ulsa
		// is an appropriate UDP multicast address prefix.
		// This makes it possible for a single UDP listener
		// to join multiple different group addresses, for
		// multiple UDP listeners that listen on the same UDP
		// port to join the same group address.
		if ulsa, err = listenerSockaddr(s, f, ulsa, toAddr); err != nil {
			closesocket(s)
			return nil, err
		}
	}

	if ulsa != nil {
		if err = syscall.Bind(s, ulsa); err != nil {
			closesocket(s)
			return nil, err
		}
	}

	if fd, err = newFD(s, f, t, net); err != nil {
		closesocket(s)
		return nil, err
	}

	// This socket is used by a dialer.
	if ursa != nil {
		if !deadline.IsZero() {
			setWriteDeadline(fd, deadline)
		}
		if err = fd.connect(ulsa, ursa); err != nil {
			fd.Close()
			return nil, err
		}
		fd.isConnected = true
		if !deadline.IsZero() {
			setWriteDeadline(fd, time.Time{})
		}
	}

	lsa, _ := syscall.Getsockname(s)
	laddr := toAddr(lsa)
	rsa, _ := syscall.Getpeername(s)
	if rsa == nil {
		rsa = ursa
	}
	raddr := toAddr(rsa)
	fd.setAddr(laddr, raddr)
	return fd, nil
}

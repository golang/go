// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// Sockets

package net

import (
	"io"
	"syscall"
)

var listenerBacklog = maxListenerBacklog()

// Generic socket creation.
func socket(net string, f, t, p int, ipv6only bool, ulsa, ursa syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err error) {
	// See ../syscall/exec_unix.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err := syscall.Socket(f, t, p)
	if err != nil {
		syscall.ForkLock.RUnlock()
		return nil, err
	}
	syscall.CloseOnExec(s)
	syscall.ForkLock.RUnlock()

	if err = setDefaultSockopts(s, f, t, ipv6only); err != nil {
		closesocket(s)
		return nil, err
	}

	var blsa syscall.Sockaddr
	if ulsa != nil {
		if blsa, err = listenerSockaddr(s, f, ulsa, toAddr); err != nil {
			closesocket(s)
			return nil, err
		}
		if err = syscall.Bind(s, blsa); err != nil {
			closesocket(s)
			return nil, err
		}
	}

	if fd, err = newFD(s, f, t, net); err != nil {
		closesocket(s)
		return nil, err
	}

	if ursa != nil {
		if err = fd.connect(ursa); err != nil {
			closesocket(s)
			fd.Close()
			return nil, err
		}
		fd.isConnected = true
	}

	lsa, _ := syscall.Getsockname(s)
	var laddr Addr
	if ulsa != nil && blsa != ulsa {
		laddr = toAddr(ulsa)
	} else {
		laddr = toAddr(lsa)
	}
	rsa, _ := syscall.Getpeername(s)
	raddr := toAddr(rsa)
	fd.setAddr(laddr, raddr)
	return fd, nil
}

type writerOnly struct {
	io.Writer
}

// Fallback implementation of io.ReaderFrom's ReadFrom, when sendfile isn't
// applicable.
func genericReadFrom(w io.Writer, r io.Reader) (n int64, err error) {
	// Use wrapper to hide existing r.ReadFrom from io.Copy.
	return io.Copy(writerOnly{w}, r)
}

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
func socket(net string, f, t, p int, la, ra syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err error) {
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err := syscall.Socket(f, t, p)
	if err != nil {
		syscall.ForkLock.RUnlock()
		return nil, err
	}
	syscall.CloseOnExec(s)
	syscall.ForkLock.RUnlock()

	err = setDefaultSockopts(s, f, t)
	if err != nil {
		closesocket(s)
		return nil, err
	}

	var bla syscall.Sockaddr
	if la != nil {
		bla, err = listenerSockaddr(s, f, la, toAddr)
		if err != nil {
			closesocket(s)
			return nil, err
		}
		err = syscall.Bind(s, bla)
		if err != nil {
			closesocket(s)
			return nil, err
		}
	}

	if fd, err = newFD(s, f, t, net); err != nil {
		closesocket(s)
		return nil, err
	}

	if ra != nil {
		if err = fd.connect(ra); err != nil {
			closesocket(s)
			fd.Close()
			return nil, err
		}
		fd.isConnected = true
	}

	sa, _ := syscall.Getsockname(s)
	var laddr Addr
	if la != nil && bla != la {
		laddr = toAddr(la)
	} else {
		laddr = toAddr(sa)
	}
	sa, _ = syscall.Getpeername(s)
	raddr := toAddr(sa)

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

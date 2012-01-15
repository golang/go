// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// Sockets

package net

import (
	"io"
	"reflect"
	"syscall"
)

var listenerBacklog = maxListenerBacklog()

// Generic socket creation.
func socket(net string, f, p, t int, la, ra syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err error) {
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, e := syscall.Socket(f, p, t)
	if err != nil {
		syscall.ForkLock.RUnlock()
		return nil, err
	}
	syscall.CloseOnExec(s)
	syscall.ForkLock.RUnlock()

	setDefaultSockopts(s, f, p)

	if la != nil {
		e = syscall.Bind(s, la)
		if e != nil {
			closesocket(s)
			return nil, e
		}
	}

	if fd, err = newFD(s, f, p, net); err != nil {
		closesocket(s)
		return nil, err
	}

	if ra != nil {
		if err = fd.connect(ra); err != nil {
			closesocket(s)
			fd.Close()
			return nil, err
		}
	}

	sa, _ := syscall.Getsockname(s)
	laddr := toAddr(sa)
	sa, _ = syscall.Getpeername(s)
	raddr := toAddr(sa)

	fd.setAddr(laddr, raddr)
	return fd, nil
}

type UnknownSocketError struct {
	sa syscall.Sockaddr
}

func (e *UnknownSocketError) Error() string {
	return "unknown socket address type " + reflect.TypeOf(e.sa).String()
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

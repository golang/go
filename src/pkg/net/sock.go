// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd windows

// Sockets

package net

import (
	"io"
	"os"
	"reflect"
	"syscall"
)

// Boolean to int.
func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Generic socket creation.
func socket(net string, f, p, t int, la, ra syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err error) {
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, e := syscall.Socket(f, p, t)
	if e != 0 {
		syscall.ForkLock.RUnlock()
		return nil, os.Errno(e)
	}
	syscall.CloseOnExec(s)
	syscall.ForkLock.RUnlock()

	setKernelSpecificSockopt(s, f)

	if la != nil {
		e = syscall.Bind(s, la)
		if e != 0 {
			closesocket(s)
			return nil, os.Errno(e)
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

func setsockoptInt(fd *netFD, level, opt int, value int) error {
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, level, opt, value))
}

func setsockoptNsec(fd *netFD, level, opt int, nsec int64) error {
	var tv = syscall.NsecToTimeval(nsec)
	return os.NewSyscallError("setsockopt", syscall.SetsockoptTimeval(fd.sysfd, level, opt, &tv))
}

func setReadBuffer(fd *netFD, bytes int) error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, bytes)
}

func setWriteBuffer(fd *netFD, bytes int) error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_SNDBUF, bytes)
}

func setReadTimeout(fd *netFD, nsec int64) error {
	fd.rdeadline_delta = nsec
	return nil
}

func setWriteTimeout(fd *netFD, nsec int64) error {
	fd.wdeadline_delta = nsec
	return nil
}

func setTimeout(fd *netFD, nsec int64) error {
	if e := setReadTimeout(fd, nsec); e != nil {
		return e
	}
	return setWriteTimeout(fd, nsec)
}

func setReuseAddr(fd *netFD, reuse bool) error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, boolint(reuse))
}

func bindToDevice(fd *netFD, dev string) error {
	// TODO(rsc): call setsockopt with null-terminated string pointer
	return os.EINVAL
}

func setDontRoute(fd *netFD, dontroute bool) error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_DONTROUTE, boolint(dontroute))
}

func setKeepAlive(fd *netFD, keepalive bool) error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, boolint(keepalive))
}

func setNoDelay(fd *netFD, noDelay bool) error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd, syscall.IPPROTO_TCP, syscall.TCP_NODELAY, boolint(noDelay))
}

func setLinger(fd *netFD, sec int) error {
	var l syscall.Linger
	if sec >= 0 {
		l.Onoff = 1
		l.Linger = int32(sec)
	} else {
		l.Onoff = 0
		l.Linger = 0
	}
	fd.incref()
	defer fd.decref()
	e := syscall.SetsockoptLinger(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_LINGER, &l)
	return os.NewSyscallError("setsockopt", e)
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

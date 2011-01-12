// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Sockets

package net

import (
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
func socket(net string, f, p, t int, la, ra syscall.Sockaddr, toAddr func(syscall.Sockaddr) Addr) (fd *netFD, err os.Error) {
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, e := syscall.Socket(f, p, t)
	if e != 0 {
		syscall.ForkLock.RUnlock()
		return nil, os.Errno(e)
	}
	syscall.CloseOnExec(s)
	syscall.ForkLock.RUnlock()

	// Allow reuse of recently-used addresses.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1)

	// Allow broadcast.
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_BROADCAST, 1)

	if f == syscall.AF_INET6 {
		// using ip, tcp, udp, etc.
		// allow both protocols even if the OS default is otherwise.
		syscall.SetsockoptInt(s, syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY, 0)
	}

	if la != nil {
		e = syscall.Bind(s, la)
		if e != 0 {
			closesocket(s)
			return nil, os.Errno(e)
		}
	}

	if ra != nil {
		e = syscall.Connect(s, ra)
		if e != 0 {
			closesocket(s)
			return nil, os.Errno(e)
		}
	}

	sa, _ := syscall.Getsockname(s)
	laddr := toAddr(sa)
	sa, _ = syscall.Getpeername(s)
	raddr := toAddr(sa)

	fd, err = newFD(s, f, p, net, laddr, raddr)
	if err != nil {
		closesocket(s)
		return nil, err
	}

	return fd, nil
}

func setsockoptInt(fd, level, opt int, value int) os.Error {
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd, level, opt, value))
}

func setsockoptNsec(fd, level, opt int, nsec int64) os.Error {
	var tv = syscall.NsecToTimeval(nsec)
	return os.NewSyscallError("setsockopt", syscall.SetsockoptTimeval(fd, level, opt, &tv))
}

func setReadBuffer(fd *netFD, bytes int) os.Error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, bytes)
}

func setWriteBuffer(fd *netFD, bytes int) os.Error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_SNDBUF, bytes)
}

func setReadTimeout(fd *netFD, nsec int64) os.Error {
	fd.rdeadline_delta = nsec
	return nil
}

func setWriteTimeout(fd *netFD, nsec int64) os.Error {
	fd.wdeadline_delta = nsec
	return nil
}

func setTimeout(fd *netFD, nsec int64) os.Error {
	if e := setReadTimeout(fd, nsec); e != nil {
		return e
	}
	return setWriteTimeout(fd, nsec)
}

func setReuseAddr(fd *netFD, reuse bool) os.Error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, boolint(reuse))
}

func bindToDevice(fd *netFD, dev string) os.Error {
	// TODO(rsc): call setsockopt with null-terminated string pointer
	return os.EINVAL
}

func setDontRoute(fd *netFD, dontroute bool) os.Error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_DONTROUTE, boolint(dontroute))
}

func setKeepAlive(fd *netFD, keepalive bool) os.Error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, boolint(keepalive))
}

func setNoDelay(fd *netFD, noDelay bool) os.Error {
	fd.incref()
	defer fd.decref()
	return setsockoptInt(fd.sysfd, syscall.IPPROTO_TCP, syscall.TCP_NODELAY, boolint(noDelay))
}

func setLinger(fd *netFD, sec int) os.Error {
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

func (e *UnknownSocketError) String() string {
	return "unknown socket address type " + reflect.Typeof(e.sa).String()
}

func sockaddrToString(sa syscall.Sockaddr) (name string, err os.Error) {
	switch a := sa.(type) {
	case *syscall.SockaddrInet4:
		return joinHostPort(IP(a.Addr[0:]).String(), itoa(a.Port)), nil
	case *syscall.SockaddrInet6:
		return joinHostPort(IP(a.Addr[0:]).String(), itoa(a.Port)), nil
	case *syscall.SockaddrUnix:
		return a.Name, nil
	}

	return "", &UnknownSocketError{sa}
}

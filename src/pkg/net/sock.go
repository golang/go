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
	// See ../syscall/exec.go for description of ForkLock.
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

	var laddr Addr
	if ulsa != nil && blsa != ulsa {
		laddr = toAddr(ulsa)
	} else {
		laddr = localSockname(fd, toAddr)
	}
	fd.setAddr(laddr, remoteSockname(fd, toAddr))
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

func localSockname(fd *netFD, toAddr func(syscall.Sockaddr) Addr) Addr {
	sa, _ := syscall.Getsockname(fd.sysfd)
	if sa == nil {
		return nullProtocolAddr(fd.family, fd.sotype)
	}
	return toAddr(sa)
}

func remoteSockname(fd *netFD, toAddr func(syscall.Sockaddr) Addr) Addr {
	sa, _ := syscall.Getpeername(fd.sysfd)
	if sa == nil {
		return nullProtocolAddr(fd.family, fd.sotype)
	}
	return toAddr(sa)
}

func nullProtocolAddr(f, t int) Addr {
	switch f {
	case syscall.AF_INET, syscall.AF_INET6:
		switch t {
		case syscall.SOCK_STREAM:
			return (*TCPAddr)(nil)
		case syscall.SOCK_DGRAM:
			return (*UDPAddr)(nil)
		case syscall.SOCK_RAW:
			return (*IPAddr)(nil)
		}
	case syscall.AF_UNIX:
		switch t {
		case syscall.SOCK_STREAM, syscall.SOCK_DGRAM, syscall.SOCK_SEQPACKET:
			return (*UnixAddr)(nil)
		}
	}
	panic("unreachable")
}

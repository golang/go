// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || windows

package net

import (
	"internal/poll"
	"os"
	"syscall"
)

func newFileFD(f *os.File) (*netFD, error) {
	s, err := dupFileSocket(f)
	if err != nil {
		return nil, err
	}
	family := syscall.AF_UNSPEC
	sotype, err := syscall.GetsockoptInt(s, syscall.SOL_SOCKET, _SO_TYPE)
	if err != nil {
		poll.CloseFunc(s)
		return nil, os.NewSyscallError("getsockopt", err)
	}
	lsa, _ := syscall.Getsockname(s)
	rsa, _ := syscall.Getpeername(s)
	switch lsa.(type) {
	case *syscall.SockaddrInet4:
		family = syscall.AF_INET
	case *syscall.SockaddrInet6:
		family = syscall.AF_INET6
	case *syscall.SockaddrUnix:
		family = syscall.AF_UNIX
	default:
		poll.CloseFunc(s)
		return nil, syscall.EPROTONOSUPPORT
	}
	fd, err := newFD(s, family, sotype, "")
	if err != nil {
		poll.CloseFunc(s)
		return nil, err
	}
	laddr := fd.addrFunc()(lsa)
	raddr := fd.addrFunc()(rsa)
	fd.net = laddr.Network()
	if err := fd.init(); err != nil {
		fd.Close()
		return nil, err
	}
	fd.setAddr(laddr, raddr)
	return fd, nil
}

func fileConn(f *os.File) (Conn, error) {
	fd, err := newFileFD(f)
	if err != nil {
		return nil, err
	}
	switch fd.laddr.(type) {
	case *TCPAddr:
		return newTCPConn(fd, defaultTCPKeepAliveIdle, KeepAliveConfig{}, testPreHookSetKeepAlive, testHookSetKeepAlive), nil
	case *UDPAddr:
		return newUDPConn(fd), nil
	case *IPAddr:
		return newIPConn(fd), nil
	case *UnixAddr:
		return newUnixConn(fd), nil
	}
	fd.Close()
	return nil, syscall.EINVAL
}

func fileListener(f *os.File) (Listener, error) {
	fd, err := newFileFD(f)
	if err != nil {
		return nil, err
	}
	switch laddr := fd.laddr.(type) {
	case *TCPAddr:
		return &TCPListener{fd: fd}, nil
	case *UnixAddr:
		return &UnixListener{fd: fd, path: laddr.Name, unlink: false}, nil
	}
	fd.Close()
	return nil, syscall.EINVAL
}

func filePacketConn(f *os.File) (PacketConn, error) {
	fd, err := newFileFD(f)
	if err != nil {
		return nil, err
	}
	switch fd.laddr.(type) {
	case *UDPAddr:
		return newUDPConn(fd), nil
	case *IPAddr:
		return newIPConn(fd), nil
	case *UnixAddr:
		return newUnixConn(fd), nil
	}
	fd.Close()
	return nil, syscall.EINVAL
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"internal/syscall/unix"
	"os"
	"syscall"
)

func dupSocket(f *os.File) (int, error) {
	s, err := dupCloseOnExec(int(f.Fd()))
	if err != nil {
		return -1, err
	}
	if err := syscall.SetNonblock(s, true); err != nil {
		closeFunc(s)
		return -1, os.NewSyscallError("setnonblock", err)
	}
	return s, nil
}

func newFileFD(f *os.File, sa SocketAddr) (*netFD, error) {
	s, err := dupSocket(f)
	if err != nil {
		return nil, err
	}
	var laddr, raddr Addr
	var fd *netFD
	if sa != nil {
		lsa := make([]byte, syscall.SizeofSockaddrAny)
		if err := unix.Getsockname(s, lsa); err != nil {
			lsa = nil
		}
		rsa := make([]byte, syscall.SizeofSockaddrAny)
		if err := unix.Getpeername(s, rsa); err != nil {
			rsa = nil
		}
		laddr = sa.Addr(lsa)
		raddr = sa.Addr(rsa)
		fd, err = newFD(s, -1, -1, laddr.Network())
	} else {
		family := syscall.AF_UNSPEC
		var sotype int
		sotype, err = syscall.GetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_TYPE)
		if err != nil {
			closeFunc(s)
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
			closeFunc(s)
			return nil, syscall.EPROTONOSUPPORT
		}
		fd, err = newFD(s, family, sotype, "")
		laddr = fd.addrFunc()(lsa)
		raddr = fd.addrFunc()(rsa)
		fd.net = laddr.Network()
	}
	if err != nil {
		closeFunc(s)
		return nil, err
	}
	if err := fd.init(); err != nil {
		fd.Close()
		return nil, err
	}
	fd.setAddr(laddr, raddr)
	return fd, nil
}

func fileConn(f *os.File) (Conn, error) {
	fd, err := newFileFD(f, nil)
	if err != nil {
		return nil, err
	}
	switch fd.laddr.(type) {
	case *TCPAddr:
		return newTCPConn(fd), nil
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
	fd, err := newFileFD(f, nil)
	if err != nil {
		return nil, err
	}
	switch laddr := fd.laddr.(type) {
	case *TCPAddr:
		return &TCPListener{fd}, nil
	case *UnixAddr:
		return &UnixListener{fd, laddr.Name}, nil
	}
	fd.Close()
	return nil, syscall.EINVAL
}

func filePacketConn(f *os.File) (PacketConn, error) {
	fd, err := newFileFD(f, nil)
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

func socketConn(f *os.File, sa SocketAddr) (Conn, error) {
	fd, err := newFileFD(f, sa)
	if err != nil {
		return nil, err
	}
	return &socketFile{conn: conn{fd}, SocketAddr: sa}, nil
}

func socketPacketConn(f *os.File, sa SocketAddr) (PacketConn, error) {
	fd, err := newFileFD(f, sa)
	if err != nil {
		return nil, err
	}
	return &socketFile{conn: conn{fd}, SocketAddr: sa}, nil
}

var (
	_ Conn       = &socketFile{}
	_ PacketConn = &socketFile{}
)

// A socketFile is a placeholder that holds a user-specified socket
// descriptor and a profile of socket address encoding.
// It implements both Conn and PacketConn interfaces.
type socketFile struct {
	conn
	SocketAddr
}

func (c *socketFile) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	from := make([]byte, syscall.SizeofSockaddrAny)
	n, err := c.fd.recvFrom(b, 0, from)
	if err != nil {
		return n, nil, &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, c.SocketAddr.Addr(from), nil
}

func (c *socketFile) WriteTo(b []byte, addr Addr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.fd.sendTo(b, 0, c.SocketAddr.Raw(addr))
	if err != nil {
		return n, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, nil
}

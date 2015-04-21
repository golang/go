// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"os"
	"syscall"
)

func newFileFD(f *os.File) (*netFD, error) {
	fd, err := dupCloseOnExec(int(f.Fd()))
	if err != nil {
		return nil, err
	}

	if err = syscall.SetNonblock(fd, true); err != nil {
		closeFunc(fd)
		return nil, os.NewSyscallError("setnonblock", err)
	}

	sotype, err := syscall.GetsockoptInt(fd, syscall.SOL_SOCKET, syscall.SO_TYPE)
	if err != nil {
		closeFunc(fd)
		return nil, os.NewSyscallError("getsockopt", err)
	}

	family := syscall.AF_UNSPEC
	toAddr := sockaddrToTCP
	lsa, _ := syscall.Getsockname(fd)
	switch lsa.(type) {
	case *syscall.SockaddrInet4:
		family = syscall.AF_INET
		if sotype == syscall.SOCK_DGRAM {
			toAddr = sockaddrToUDP
		} else if sotype == syscall.SOCK_RAW {
			toAddr = sockaddrToIP
		}
	case *syscall.SockaddrInet6:
		family = syscall.AF_INET6
		if sotype == syscall.SOCK_DGRAM {
			toAddr = sockaddrToUDP
		} else if sotype == syscall.SOCK_RAW {
			toAddr = sockaddrToIP
		}
	case *syscall.SockaddrUnix:
		family = syscall.AF_UNIX
		toAddr = sockaddrToUnix
		if sotype == syscall.SOCK_DGRAM {
			toAddr = sockaddrToUnixgram
		} else if sotype == syscall.SOCK_SEQPACKET {
			toAddr = sockaddrToUnixpacket
		}
	default:
		closeFunc(fd)
		return nil, syscall.EPROTONOSUPPORT
	}
	laddr := toAddr(lsa)
	rsa, _ := syscall.Getpeername(fd)
	raddr := toAddr(rsa)

	netfd, err := newFD(fd, family, sotype, laddr.Network())
	if err != nil {
		closeFunc(fd)
		return nil, err
	}
	if err := netfd.init(); err != nil {
		netfd.Close()
		return nil, err
	}
	netfd.setAddr(laddr, raddr)
	return netfd, nil
}

func fileConn(f *os.File) (Conn, error) {
	fd, err := newFileFD(f)
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
	fd, err := newFileFD(f)
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

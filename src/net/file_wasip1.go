// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package net

import (
	"os"
	"syscall"
	_ "unsafe" // for go:linkname
)

func fileListener(f *os.File) (Listener, error) {
	filetype, err := fd_fdstat_get_type(f.PollFD().Sysfd)
	if err != nil {
		return nil, err
	}
	net, err := fileListenNet(filetype)
	if err != nil {
		return nil, err
	}
	pfd := f.PollFD().Copy()
	fd := newPollFD(net, pfd)
	if err := fd.init(); err != nil {
		pfd.Close()
		return nil, err
	}
	return newFileListener(fd), nil
}

func fileConn(f *os.File) (Conn, error) {
	filetype, err := fd_fdstat_get_type(f.PollFD().Sysfd)
	if err != nil {
		return nil, err
	}
	net, err := fileConnNet(filetype)
	if err != nil {
		return nil, err
	}
	pfd := f.PollFD().Copy()
	fd := newPollFD(net, pfd)
	if err := fd.init(); err != nil {
		pfd.Close()
		return nil, err
	}
	return newFileConn(fd), nil
}

func filePacketConn(f *os.File) (PacketConn, error) {
	return nil, syscall.ENOPROTOOPT
}

func fileListenNet(filetype syscall.Filetype) (string, error) {
	switch filetype {
	case syscall.FILETYPE_SOCKET_STREAM:
		return "tcp", nil
	case syscall.FILETYPE_SOCKET_DGRAM:
		return "", syscall.EOPNOTSUPP
	default:
		return "", syscall.ENOTSOCK
	}
}

func fileConnNet(filetype syscall.Filetype) (string, error) {
	switch filetype {
	case syscall.FILETYPE_SOCKET_STREAM:
		return "tcp", nil
	case syscall.FILETYPE_SOCKET_DGRAM:
		return "udp", nil
	default:
		return "", syscall.ENOTSOCK
	}
}

func newFileListener(fd *netFD) Listener {
	switch fd.net {
	case "tcp":
		return &TCPListener{fd: fd}
	default:
		panic("unsupported network for file listener: " + fd.net)
	}
}

func newFileConn(fd *netFD) Conn {
	switch fd.net {
	case "tcp":
		return &TCPConn{conn{fd: fd}}
	case "udp":
		return &UDPConn{conn{fd: fd}}
	default:
		panic("unsupported network for file connection: " + fd.net)
	}
}

// This helper is implemented in the syscall package. It means we don't have
// to redefine the fd_fdstat_get host import or the fdstat struct it
// populates.
//
//go:linkname fd_fdstat_get_type syscall.fd_fdstat_get_type
func fd_fdstat_get_type(fd int) (uint8, error)

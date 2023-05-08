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
	fd, err := newFileFD(f)
	if err != nil {
		return nil, err
	}
	return &TCPListener{fd: fd}, nil
}

func fileConn(f *os.File) (Conn, error) {
	fd, err := newFileFD(f)
	if err != nil {
		return nil, err
	}
	return &TCPConn{conn{fd: fd}}, nil
}

func filePacketConn(f *os.File) (PacketConn, error) { return nil, syscall.ENOPROTOOPT }

func newFileFD(f *os.File) (fd *netFD, err error) {
	pfd := f.PollFD().Copy()
	defer func() {
		if err != nil {
			pfd.Close()
		}
	}()
	filetype, err := fd_fdstat_get_type(pfd.Sysfd)
	if err != nil {
		return nil, err
	}
	if filetype != syscall.FILETYPE_SOCKET_STREAM {
		return nil, syscall.ENOTSOCK
	}
	fd, err = newPollFD(pfd)
	if err != nil {
		return nil, err
	}
	if err := fd.init(); err != nil {
		return nil, err
	}
	return fd, nil
}

// This helper is implemented in the syscall package. It means we don't have
// to redefine the fd_fdstat_get host import or the fdstat struct it
// populates.
//
//go:linkname fd_fdstat_get_type syscall.fd_fdstat_get_type
func fd_fdstat_get_type(fd int) (uint8, error)

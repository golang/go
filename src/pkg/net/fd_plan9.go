// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"syscall"
	"time"
)

// Network file descritor.
type netFD struct {
	proto, name, dir string
	ctl, data        *os.File
	laddr, raddr     Addr
}

var canCancelIO = true // used for testing current package

func sysInit() {
}

func dialTimeout(net, addr string, timeout time.Duration) (Conn, error) {
	// On plan9, use the relatively inefficient
	// goroutine-racing implementation.
	return dialTimeoutRace(net, addr, timeout)
}

func newFD(proto, name string, ctl *os.File, laddr, raddr Addr) *netFD {
	return &netFD{proto, name, "/net/" + proto + "/" + name, ctl, nil, laddr, raddr}
}

func (fd *netFD) ok() bool { return fd != nil && fd.ctl != nil }

func (fd *netFD) Read(b []byte) (n int, err error) {
	if !fd.ok() {
		return 0, syscall.EINVAL
	}
	if fd.data == nil {
		fd.data, err = os.OpenFile(fd.dir+"/data", os.O_RDWR, 0)
		if err != nil {
			return 0, err
		}
	}
	n, err = fd.data.Read(b)
	if fd.proto == "udp" && err == io.EOF {
		n = 0
		err = nil
	}
	return
}

func (fd *netFD) Write(b []byte) (n int, err error) {
	if !fd.ok() {
		return 0, syscall.EINVAL
	}
	if fd.data == nil {
		fd.data, err = os.OpenFile(fd.dir+"/data", os.O_RDWR, 0)
		if err != nil {
			return 0, err
		}
	}
	return fd.data.Write(b)
}

func (fd *netFD) CloseRead() error {
	if !fd.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

func (fd *netFD) CloseWrite() error {
	if !fd.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

func (fd *netFD) Close() error {
	if !fd.ok() {
		return syscall.EINVAL
	}
	err := fd.ctl.Close()
	if err != nil {
		return err
	}
	if fd.data != nil {
		err = fd.data.Close()
	}
	fd.ctl = nil
	fd.data = nil
	return err
}

func (fd *netFD) dup() (*os.File, error) {
	return nil, syscall.EPLAN9
}

func setDeadline(fd *netFD, t time.Time) error {
	return syscall.EPLAN9
}

func setReadDeadline(fd *netFD, t time.Time) error {
	return syscall.EPLAN9
}

func setWriteDeadline(fd *netFD, t time.Time) error {
	return syscall.EPLAN9
}

func setReadBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

func setWriteBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

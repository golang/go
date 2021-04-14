// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"io"
	"os"
	"syscall"
	"time"
)

// Network file descriptor.
type netFD struct {
	pfd poll.FD

	// immutable until Close
	net               string
	n                 string
	dir               string
	listen, ctl, data *os.File
	laddr, raddr      Addr
	isStream          bool
}

var netdir = "/net" // default network

func newFD(net, name string, listen, ctl, data *os.File, laddr, raddr Addr) (*netFD, error) {
	ret := &netFD{
		net:    net,
		n:      name,
		dir:    netdir + "/" + net + "/" + name,
		listen: listen,
		ctl:    ctl, data: data,
		laddr: laddr,
		raddr: raddr,
	}
	ret.pfd.Destroy = ret.destroy
	return ret, nil
}

func (fd *netFD) init() error {
	// stub for future fd.pd.Init(fd)
	return nil
}

func (fd *netFD) name() string {
	var ls, rs string
	if fd.laddr != nil {
		ls = fd.laddr.String()
	}
	if fd.raddr != nil {
		rs = fd.raddr.String()
	}
	return fd.net + ":" + ls + "->" + rs
}

func (fd *netFD) ok() bool { return fd != nil && fd.ctl != nil }

func (fd *netFD) destroy() {
	if !fd.ok() {
		return
	}
	err := fd.ctl.Close()
	if fd.data != nil {
		if err1 := fd.data.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	if fd.listen != nil {
		if err1 := fd.listen.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	fd.ctl = nil
	fd.data = nil
	fd.listen = nil
}

func (fd *netFD) Read(b []byte) (n int, err error) {
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
	}
	n, err = fd.pfd.Read(fd.data.Read, b)
	if fd.net == "udp" && err == io.EOF {
		n = 0
		err = nil
	}
	return
}

func (fd *netFD) Write(b []byte) (n int, err error) {
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
	}
	return fd.pfd.Write(fd.data.Write, b)
}

func (fd *netFD) closeRead() error {
	if !fd.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

func (fd *netFD) closeWrite() error {
	if !fd.ok() {
		return syscall.EINVAL
	}
	return syscall.EPLAN9
}

func (fd *netFD) Close() error {
	if err := fd.pfd.Close(); err != nil {
		return err
	}
	if !fd.ok() {
		return syscall.EINVAL
	}
	if fd.net == "tcp" {
		// The following line is required to unblock Reads.
		_, err := fd.ctl.WriteString("close")
		if err != nil {
			return err
		}
	}
	err := fd.ctl.Close()
	if fd.data != nil {
		if err1 := fd.data.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	if fd.listen != nil {
		if err1 := fd.listen.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	fd.ctl = nil
	fd.data = nil
	fd.listen = nil
	return err
}

// This method is only called via Conn.
func (fd *netFD) dup() (*os.File, error) {
	if !fd.ok() || fd.data == nil {
		return nil, syscall.EINVAL
	}
	return fd.file(fd.data, fd.dir+"/data")
}

func (l *TCPListener) dup() (*os.File, error) {
	if !l.fd.ok() {
		return nil, syscall.EINVAL
	}
	return l.fd.file(l.fd.ctl, l.fd.dir+"/ctl")
}

func (fd *netFD) file(f *os.File, s string) (*os.File, error) {
	dfd, err := syscall.Dup(int(f.Fd()), -1)
	if err != nil {
		return nil, os.NewSyscallError("dup", err)
	}
	return os.NewFile(uintptr(dfd), s), nil
}

func setReadBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

func setWriteBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

func (fd *netFD) SetDeadline(t time.Time) error {
	return fd.pfd.SetDeadline(t)
}

func (fd *netFD) SetReadDeadline(t time.Time) error {
	return fd.pfd.SetReadDeadline(t)
}

func (fd *netFD) SetWriteDeadline(t time.Time) error {
	return fd.pfd.SetWriteDeadline(t)
}

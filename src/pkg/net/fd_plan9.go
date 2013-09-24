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

func sysInit() {
}

func dial(net string, ra Addr, dialer func(time.Time) (Conn, error), deadline time.Time) (Conn, error) {
	// On plan9, use the relatively inefficient
	// goroutine-racing implementation.
	return dialChannel(net, ra, dialer, deadline)
}

func newFD(proto, name string, ctl, data *os.File, laddr, raddr Addr) *netFD {
	return &netFD{proto, name, "/net/" + proto + "/" + name, ctl, data, laddr, raddr}
}

func (fd *netFD) ok() bool { return fd != nil && fd.ctl != nil }

func (fd *netFD) Read(b []byte) (n int, err error) {
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
	}
	n, err = fd.data.Read(b)
	if fd.proto == "udp" && err == io.EOF {
		n = 0
		err = nil
	}
	return
}

func (fd *netFD) Write(b []byte) (n int, err error) {
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
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
	if fd.data != nil {
		if err1 := fd.data.Close(); err1 != nil && err == nil {
			err = err1
		}
	}
	fd.ctl = nil
	fd.data = nil
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
	syscall.ForkLock.RLock()
	dfd, err := syscall.Dup(int(f.Fd()), -1)
	syscall.ForkLock.RUnlock()
	if err != nil {
		return nil, &OpError{"dup", s, fd.laddr, err}
	}
	return os.NewFile(uintptr(dfd), s), nil
}

func (fd *netFD) setDeadline(t time.Time) error {
	return syscall.EPLAN9
}

func (fd *netFD) setReadDeadline(t time.Time) error {
	return syscall.EPLAN9
}

func (fd *netFD) setWriteDeadline(t time.Time) error {
	return syscall.EPLAN9
}

func setReadBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

func setWriteBuffer(fd *netFD, bytes int) error {
	return syscall.EPLAN9
}

func skipRawSocketTests() (skip bool, skipmsg string, err error) {
	return true, "skipping test on plan9", nil
}

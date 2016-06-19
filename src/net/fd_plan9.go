// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"syscall"
	"time"
)

// Network file descriptor.
type netFD struct {
	// locking/lifetime of sysfd + serialize access to Read and Write methods
	fdmu fdMutex

	// immutable until Close
	net          string
	n            string
	dir          string
	ctl, data    *os.File
	laddr, raddr Addr
}

var (
	netdir string // default network
)

func sysInit() {
	netdir = "/net"
}

func newFD(net, name string, ctl, data *os.File, laddr, raddr Addr) (*netFD, error) {
	return &netFD{net: net, n: name, dir: netdir + "/" + net + "/" + name, ctl: ctl, data: data, laddr: laddr, raddr: raddr}, nil
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
	fd.ctl = nil
	fd.data = nil
}

func (fd *netFD) Read(b []byte) (n int, err error) {
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
	}
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if len(b) == 0 {
		return 0, nil
	}
	n, err = fd.data.Read(b)
	if isHangup(err) {
		err = io.EOF
	}
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
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	return fd.data.Write(b)
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
	if !fd.fdmu.increfAndClose() {
		return errClosing
	}
	if !fd.ok() {
		return syscall.EINVAL
	}
	if fd.net == "tcp" {
		// The following line is required to unblock Reads.
		// For some reason, WriteString returns an error:
		// "write /net/tcp/39/listen: inappropriate use of fd"
		// But without it, Reads on dead conns hang forever.
		// See Issue 9554.
		fd.ctl.WriteString("hangup")
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
	dfd, err := syscall.Dup(int(f.Fd()), -1)
	if err != nil {
		return nil, os.NewSyscallError("dup", err)
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

func isHangup(err error) bool {
	return err != nil && stringsHasSuffix(err.Error(), "Hangup")
}

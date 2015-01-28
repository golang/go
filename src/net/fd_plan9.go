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
	// locking/lifetime of sysfd + serialize access to Read and Write methods
	fdmu fdMutex

	// immutable until Close
	proto        string
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

func dial(net string, ra Addr, dialer func(time.Time) (Conn, error), deadline time.Time) (Conn, error) {
	// On plan9, use the relatively inefficient
	// goroutine-racing implementation.
	return dialChannel(net, ra, dialer, deadline)
}

func newFD(proto, name string, ctl, data *os.File, laddr, raddr Addr) (*netFD, error) {
	return &netFD{proto: proto, n: name, dir: netdir + "/" + proto + "/" + name, ctl: ctl, data: data, laddr: laddr, raddr: raddr}, nil
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
	return fd.proto + ":" + ls + "->" + rs
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

// Add a reference to this fd.
// Returns an error if the fd cannot be used.
func (fd *netFD) incref() error {
	if !fd.fdmu.Incref() {
		return errClosing
	}
	return nil
}

// Remove a reference to this FD and close if we've been asked to do so
// (and there are no references left).
func (fd *netFD) decref() {
	if fd.fdmu.Decref() {
		fd.destroy()
	}
}

// Add a reference to this fd and lock for reading.
// Returns an error if the fd cannot be used.
func (fd *netFD) readLock() error {
	if !fd.fdmu.RWLock(true) {
		return errClosing
	}
	return nil
}

// Unlock for reading and remove a reference to this FD.
func (fd *netFD) readUnlock() {
	if fd.fdmu.RWUnlock(true) {
		fd.destroy()
	}
}

// Add a reference to this fd and lock for writing.
// Returns an error if the fd cannot be used.
func (fd *netFD) writeLock() error {
	if !fd.fdmu.RWLock(false) {
		return errClosing
	}
	return nil
}

// Unlock for writing and remove a reference to this FD.
func (fd *netFD) writeUnlock() {
	if fd.fdmu.RWUnlock(false) {
		fd.destroy()
	}
}

func (fd *netFD) Read(b []byte) (n int, err error) {
	if !fd.ok() || fd.data == nil {
		return 0, syscall.EINVAL
	}
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
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
	if !fd.fdmu.IncrefAndClose() {
		return errClosing
	}
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

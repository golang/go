// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"syscall"
)

// BUG(mikio): On Windows, the Read and Write methods of
// syscall.RawConn are not implemented.

// BUG(mikio): On NaCl and Plan 9, the Control, Read and Write methods
// of syscall.RawConn are not implemented.

type rawConn struct {
	fd *netFD
}

func (c *rawConn) ok() bool { return c != nil && c.fd != nil }

func (c *rawConn) Control(f func(uintptr)) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	err := c.fd.pfd.RawControl(f)
	runtime.KeepAlive(c.fd)
	if err != nil {
		err = &OpError{Op: "raw-control", Net: c.fd.net, Source: nil, Addr: c.fd.laddr, Err: err}
	}
	return err
}

func (c *rawConn) Read(f func(uintptr) bool) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	err := c.fd.pfd.RawRead(f)
	runtime.KeepAlive(c.fd)
	if err != nil {
		err = &OpError{Op: "raw-read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return err
}

func (c *rawConn) Write(f func(uintptr) bool) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	err := c.fd.pfd.RawWrite(f)
	runtime.KeepAlive(c.fd)
	if err != nil {
		err = &OpError{Op: "raw-write", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return err
}

func newRawConn(fd *netFD) (*rawConn, error) {
	return &rawConn{fd: fd}, nil
}

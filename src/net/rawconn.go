// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"runtime"
	"syscall"
)

// BUG(tmm1): On Windows, the Write method of syscall.RawConn
// does not integrate with the runtime's network poller. It cannot
// wait for the connection to become writeable, and does not respect
// deadlines. If the user-provided callback returns false, the Write
// method will fail immediately.

// BUG(mikio): On JS and Plan 9, the Control, Read and Write
// methods of syscall.RawConn are not implemented.

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

// PollFD returns the poll.FD of the underlying connection.
//
// Other packages in std that also import [internal/poll] (such as os)
// can use a type assertion to access this extension method so that
// they can pass the *poll.FD to functions like poll.Splice.
//
// PollFD is not intended for use outside the standard library.
func (c *rawConn) PollFD() *poll.FD {
	if !c.ok() {
		return nil
	}
	return &c.fd.pfd
}

func newRawConn(fd *netFD) *rawConn {
	return &rawConn{fd: fd}
}

// Network returns the network type of the underlying connection.
//
// Other packages in std that import internal/poll and are unable to
// import net (such as os) can use a type assertion to access this
// extension method so that they can distinguish different socket types.
//
// Network is not intended for use outside the standard library.
func (c *rawConn) Network() poll.String {
	return poll.String(c.fd.net)
}

type rawListener struct {
	rawConn
}

func (l *rawListener) Read(func(uintptr) bool) error {
	return syscall.EINVAL
}

func (l *rawListener) Write(func(uintptr) bool) error {
	return syscall.EINVAL
}

func newRawListener(fd *netFD) *rawListener {
	return &rawListener{rawConn{fd: fd}}
}

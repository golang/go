// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris

package net

import (
	"runtime"
	"syscall"
)

func (c *conn) writeBuffers(v *Buffers) (int64, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.fd.writeBuffers(v)
	if err != nil {
		return n, &OpError{Op: "writev", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, nil
}

func (fd *netFD) writeBuffers(v *Buffers) (n int64, err error) {
	n, err = fd.pfd.Writev((*[][]byte)(v))
	runtime.KeepAlive(fd)
	return n, wrapSyscallError("writev", err)
}

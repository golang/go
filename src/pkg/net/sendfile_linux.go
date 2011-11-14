// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"syscall"
)

// maxSendfileSize is the largest chunk size we ask the kernel to copy
// at a time.
const maxSendfileSize int = 4 << 20

// sendFile copies the contents of r to c using the sendfile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number of bytes copied and any
// non-EOF error.
//
// if handled == false, sendFile performed no work.
func sendFile(c *netFD, r io.Reader) (written int64, err error, handled bool) {
	var remain int64 = 1 << 62 // by default, copy until EOF

	lr, ok := r.(*io.LimitedReader)
	if ok {
		remain, r = lr.N, lr.R
		if remain <= 0 {
			return 0, nil, true
		}
	}
	f, ok := r.(*os.File)
	if !ok {
		return 0, nil, false
	}

	c.wio.Lock()
	defer c.wio.Unlock()
	c.incref()
	defer c.decref()
	if c.wdeadline_delta > 0 {
		// This is a little odd that we're setting the timeout
		// for the entire file but Write has the same issue
		// (if one slurps the whole file into memory and
		// do one large Write). At least they're consistent.
		c.wdeadline = pollserver.Now() + c.wdeadline_delta
	} else {
		c.wdeadline = 0
	}

	dst := c.sysfd
	src := f.Fd()
	for remain > 0 {
		n := maxSendfileSize
		if int64(n) > remain {
			n = int(remain)
		}
		n, errno := syscall.Sendfile(dst, src, nil, n)
		if n > 0 {
			written += int64(n)
			remain -= int64(n)
		}
		if n == 0 && errno == nil {
			break
		}
		if errno == syscall.EAGAIN && c.wdeadline >= 0 {
			pollserver.WaitWrite(c)
			continue
		}
		if errno != nil {
			// This includes syscall.ENOSYS (no kernel
			// support) and syscall.EINVAL (fd types which
			// don't implement sendfile together)
			err = &OpError{"sendfile", c.net, c.raddr, errno}
			break
		}
	}
	if lr != nil {
		lr.N = remain
	}
	return written, err, written > 0
}

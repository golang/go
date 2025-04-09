// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux || (darwin && !ios) || dragonfly || freebsd || solaris || windows

package net

import (
	"internal/poll"
	"io"
	"syscall"
)

const supportsSendfile = true

// sendFile copies the contents of r to c using the sendfile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number (potentially zero) of bytes
// copied and any non-EOF error.
//
// if handled == false, sendFile performed no work.
func sendFile(c *netFD, r io.Reader) (written int64, err error, handled bool) {
	var remain int64 = 0 // 0 writes the entire file
	lr, ok := r.(*io.LimitedReader)
	if ok {
		remain, r = lr.N, lr.R
		if remain <= 0 {
			return 0, nil, true
		}
	}
	// r might be an *os.File or an os.fileWithoutWriteTo.
	// Type assert to an interface rather than *os.File directly to handle the latter case.
	f, ok := r.(syscall.Conn)
	if !ok {
		return 0, nil, false
	}

	sc, err := f.SyscallConn()
	if err != nil {
		return 0, nil, false
	}

	var werr error
	err = sc.Read(func(fd uintptr) bool {
		written, werr, handled = poll.SendFile(&c.pfd, fd, remain)
		return true
	})
	if err == nil {
		err = werr
	}

	if lr != nil {
		lr.N = remain - written
	}

	return written, wrapSyscallError("sendfile", err), handled
}

// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"io"
	"os"
	"syscall"
)

// Not strictly needed, but very helpful for debugging, see issue #10221.
//go:cgo_import_dynamic _ _ "libsendfile.so"
//go:cgo_import_dynamic _ _ "libsocket.so"

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
	// Solaris uses 0 as the "until EOF" value. If you pass in more bytes than the
	// file contains, it will loop back to the beginning ad nauseam until it's sent
	// exactly the number of bytes told to. As such, we need to know exactly how many
	// bytes to send.
	var remain int64 = 0

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

	if remain == 0 {
		fi, err := f.Stat()
		if err != nil {
			return 0, err, false
		}

		remain = fi.Size()
	}

	// The other quirk with Solaris's sendfile implementation is that it doesn't
	// use the current position of the file -- if you pass it offset 0, it starts
	// from offset 0. There's no way to tell it "start from current position", so
	// we have to manage that explicitly.
	pos, err := f.Seek(0, os.SEEK_CUR)
	if err != nil {
		return 0, err, false
	}

	if err := c.writeLock(); err != nil {
		return 0, err, true
	}
	defer c.writeUnlock()

	dst := c.sysfd
	src := int(f.Fd())
	for remain > 0 {
		n := maxSendfileSize
		if int64(n) > remain {
			n = int(remain)
		}
		pos1 := pos
		n, err1 := syscall.Sendfile(dst, src, &pos1, n)
		if n > 0 {
			pos += int64(n)
			written += int64(n)
			remain -= int64(n)
		}
		if n == 0 && err1 == nil {
			break
		}
		if err1 == syscall.EAGAIN {
			if err1 = c.pd.WaitWrite(); err1 == nil {
				continue
			}
		}
		if err1 == syscall.EINTR {
			continue
		}
		if err1 != nil {
			// This includes syscall.ENOSYS (no kernel
			// support) and syscall.EINVAL (fd types which
			// don't implement sendfile)
			err = err1
			break
		}
	}
	if lr != nil {
		lr.N = remain
	}
	if err != nil {
		err = os.NewSyscallError("sendfile", err)
	}
	return written, err, written > 0
}

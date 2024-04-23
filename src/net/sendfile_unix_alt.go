// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (darwin && !ios) || dragonfly || freebsd || solaris

package net

import (
	"internal/poll"
	"io"
	"io/fs"
	"syscall"
)

// sendFile copies the contents of r to c using the sendfile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number (potentially zero) of bytes
// copied and any non-EOF error.
//
// if handled == false, sendFile performed no work.
func sendFile(c *netFD, r io.Reader) (written int64, err error, handled bool) {
	// Darwin, FreeBSD, DragonFly and Solaris use 0 as the "until EOF" value.
	// If you pass in more bytes than the file contains, it will
	// loop back to the beginning ad nauseam until it's sent
	// exactly the number of bytes told to. As such, we need to
	// know exactly how many bytes to send.
	var remain int64 = 0

	lr, ok := r.(*io.LimitedReader)
	if ok {
		remain, r = lr.N, lr.R
		if remain <= 0 {
			return 0, nil, true
		}
	}
	f, ok := r.(interface {
		fs.File
		io.Seeker
		syscall.Conn
	})
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

	// The other quirk with Darwin/FreeBSD/DragonFly/Solaris's sendfile
	// implementation is that it doesn't use the current position
	// of the file -- if you pass it offset 0, it starts from
	// offset 0. There's no way to tell it "start from current
	// position", so we have to manage that explicitly.
	pos, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return 0, err, false
	}

	sc, err := f.SyscallConn()
	if err != nil {
		return 0, nil, false
	}

	var werr error
	err = sc.Read(func(fd uintptr) bool {
		written, werr, handled = poll.SendFile(&c.pfd, int(fd), pos, remain)
		return true
	})
	if err == nil {
		err = werr
	}

	if lr != nil {
		lr.N = remain - written
	}

	_, err1 := f.Seek(written, io.SeekCurrent)
	if err1 != nil && err == nil {
		return written, err1, handled
	}

	return written, wrapSyscallError("sendfile", err), handled
}

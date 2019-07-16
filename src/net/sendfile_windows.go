// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"io"
	"os"
	"syscall"
)

// sendFile copies the contents of r to c using the TransmitFile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number of bytes copied and any
// non-EOF error.
//
// if handled == false, sendFile performed no work.
//
// Note that sendfile for windows does not support >2GB file.
func sendFile(fd *netFD, r io.Reader) (written int64, err error, handled bool) {
	var n int64 = 0 // by default, copy until EOF

	lr, ok := r.(*io.LimitedReader)
	if ok {
		n, r = lr.N, lr.R
		if n <= 0 {
			return 0, nil, true
		}
	}
	f, ok := r.(*os.File)
	if !ok {
		return 0, nil, false
	}

	done, err := poll.SendFile(&fd.pfd, syscall.Handle(f.Fd()), n)

	if err != nil {
		return 0, wrapSyscallError("transmitfile", err), false
	}
	if lr != nil {
		lr.N -= int64(done)
	}
	return int64(done), nil, true
}

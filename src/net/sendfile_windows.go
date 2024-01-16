// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"io"
	"syscall"
)

// sendFile copies the contents of r to c using the TransmitFile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number of bytes copied and any
// non-EOF error.
//
// if handled == false, sendFile performed no work.
func sendFile(fd *netFD, r io.Reader) (written int64, err error, handled bool) {
	var n int64 = 0 // by default, copy until EOF.

	lr, ok := r.(*io.LimitedReader)
	if ok {
		n, r = lr.N, lr.R
		if n <= 0 {
			return 0, nil, true
		}
	}

	f, ok := r.(interface {
		Fd() uintptr
	})
	if !ok {
		return 0, nil, false
	}

	written, err = poll.SendFile(&fd.pfd, syscall.Handle(f.Fd()), n)
	if err != nil {
		err = wrapSyscallError("transmitfile", err)
	}

	// If any byte was copied, regardless of any error
	// encountered mid-way, handled must be set to true.
	handled = written > 0

	return
}

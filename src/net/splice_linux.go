// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"io"
)

// splice transfers data from r to c using the splice system call to minimize
// copies from and to userspace. c must be a TCP connection. Currently, splice
// is only enabled if r is also a TCP connection.
//
// If splice returns handled == false, it has performed no work.
func splice(c *netFD, r io.Reader) (written int64, err error, handled bool) {
	var remain int64 = 1 << 62 // by default, copy until EOF
	lr, ok := r.(*io.LimitedReader)
	if ok {
		remain, r = lr.N, lr.R
		if remain <= 0 {
			return 0, nil, true
		}
	}
	s, ok := r.(*TCPConn)
	if !ok {
		return 0, nil, false
	}
	written, handled, sc, err := poll.Splice(&c.pfd, &s.fd.pfd, remain)
	if lr != nil {
		lr.N -= written
	}
	return written, wrapSyscallError(sc, err), handled
}

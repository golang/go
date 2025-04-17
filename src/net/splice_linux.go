// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"io"
)

var pollSplice = poll.Splice

// spliceFrom transfers data from r to c using the splice system call to minimize
// copies from and to userspace. c must be a TCP connection.
// Currently, spliceFrom is only enabled if r is a TCP or a stream-oriented Unix connection.
//
// If spliceFrom returns handled == false, it has performed no work.
func spliceFrom(c *netFD, r io.Reader) (written int64, err error, handled bool) {
	var remain int64 = 1<<63 - 1 // by default, copy until EOF
	lr, ok := r.(*io.LimitedReader)
	if ok {
		remain, r = lr.N, lr.R
		if remain <= 0 {
			return 0, nil, true
		}
	}

	var s *netFD
	switch v := r.(type) {
	case *TCPConn:
		s = v.fd
	case tcpConnWithoutWriteTo:
		s = v.fd
	case *UnixConn:
		if v.fd.net != "unix" {
			return 0, nil, false
		}
		s = v.fd
	default:
		return 0, nil, false
	}

	written, handled, err = pollSplice(&c.pfd, &s.pfd, remain)
	if lr != nil {
		lr.N -= written
	}
	return written, wrapSyscallError("splice", err), handled
}

// spliceTo transfers data from c to w using the splice system call to minimize
// copies from and to userspace. c must be a TCP connection.
// Currently, spliceTo is only enabled if w is a stream-oriented Unix connection.
//
// If spliceTo returns handled == false, it has performed no work.
func spliceTo(w io.Writer, c *netFD) (written int64, err error, handled bool) {
	uc, ok := w.(*UnixConn)
	if !ok || uc.fd.net != "unix" {
		return
	}

	written, handled, err = pollSplice(&uc.fd.pfd, &c.pfd, 1<<63-1)
	return written, wrapSyscallError("splice", err), handled
}

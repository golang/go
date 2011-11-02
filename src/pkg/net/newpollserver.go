// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

package net

import (
	"os"
	"syscall"
)

func newPollServer() (s *pollServer, err error) {
	s = new(pollServer)
	s.cr = make(chan *netFD, 1)
	s.cw = make(chan *netFD, 1)
	if s.pr, s.pw, err = os.Pipe(); err != nil {
		return nil, err
	}
	var e int
	if e = syscall.SetNonblock(s.pr.Fd(), true); e != 0 {
		goto Errno
	}
	if e = syscall.SetNonblock(s.pw.Fd(), true); e != 0 {
		goto Errno
	}
	if s.poll, err = newpollster(); err != nil {
		goto Error
	}
	if _, err = s.poll.AddFD(s.pr.Fd(), 'r', true); err != nil {
		s.poll.Close()
		goto Error
	}
	s.pending = make(map[int]*netFD)
	go s.Run()
	return s, nil

Errno:
	err = &os.PathError{"setnonblock", s.pr.Name(), os.Errno(e)}
Error:
	s.pr.Close()
	s.pw.Close()
	return nil, err
}

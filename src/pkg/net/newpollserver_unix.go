// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package net

import (
	"os"
	"syscall"
)

func newPollServer() (s *pollServer, err error) {
	s = new(pollServer)
	if s.pr, s.pw, err = os.Pipe(); err != nil {
		return nil, err
	}
	if err = syscall.SetNonblock(int(s.pr.Fd()), true); err != nil {
		goto Errno
	}
	if err = syscall.SetNonblock(int(s.pw.Fd()), true); err != nil {
		goto Errno
	}
	if s.poll, err = newpollster(); err != nil {
		goto Error
	}
	if _, err = s.poll.AddFD(int(s.pr.Fd()), 'r', true); err != nil {
		s.poll.Close()
		goto Error
	}
	s.pending = make(map[int]*netFD)
	go s.Run()
	return s, nil

Errno:
	err = &os.PathError{
		Op:   "setnonblock",
		Path: s.pr.Name(),
		Err:  err,
	}
Error:
	s.pr.Close()
	s.pw.Close()
	return nil, err
}

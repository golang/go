// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): All the prints in this file should go to standard error.

package net

import (
	"os"
	"syscall"
)

func newPollServer() (s *pollServer, err os.Error) {
	s = new(pollServer)
	s.cr = make(chan *netFD, 1)
	s.cw = make(chan *netFD, 1)
	if s.pr, s.pw, err = os.Pipe(); err != nil {
		return nil, err
	}
	var e int
	if e = syscall.SetNonblock(s.pr.Fd(), true); e != 0 {
	Errno:
		err = &os.PathError{"setnonblock", s.pr.Name(), os.Errno(e)}
	Error:
		s.pr.Close()
		s.pw.Close()
		return nil, err
	}
	if e = syscall.SetNonblock(s.pw.Fd(), true); e != 0 {
		goto Errno
	}
	if s.poll, err = newpollster(); err != nil {
		goto Error
	}
	if err = s.poll.AddFD(s.pr.Fd(), 'r', true); err != nil {
		s.poll.Close()
		goto Error
	}
	s.pending = make(map[int]*netFD)
	go s.Run()
	return s, nil
}

func (s *pollServer) Run() {
	var scratch [100]byte
	for {
		var t = s.deadline
		if t > 0 {
			t = t - s.Now()
			if t <= 0 {
				s.CheckDeadlines()
				continue
			}
		}
		fd, mode, err := s.poll.WaitFD(t)
		if err != nil {
			print("pollServer WaitFD: ", err.String(), "\n")
			return
		}
		if fd < 0 {
			// Timeout happened.
			s.CheckDeadlines()
			continue
		}
		if fd == s.pr.Fd() {
			// Drain our wakeup pipe.
			for nn, _ := s.pr.Read(scratch[0:]); nn > 0; {
				nn, _ = s.pr.Read(scratch[0:])
			}
			// Read from channels
			for fd, ok := <-s.cr; ok; fd, ok = <-s.cr {
				s.AddFD(fd, 'r')
			}
			for fd, ok := <-s.cw; ok; fd, ok = <-s.cw {
				s.AddFD(fd, 'w')
			}
		} else {
			netfd := s.LookupFD(fd, mode)
			if netfd == nil {
				print("pollServer: unexpected wakeup for fd=", netfd, " mode=", string(mode), "\n")
				continue
			}
			s.WakeFD(netfd, mode)
		}
	}
}

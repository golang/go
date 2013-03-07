// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package net

import (
	"os"
	"runtime"
	"sync"
	"syscall"
	"time"
)

// A pollServer helps FDs determine when to retry a non-blocking
// read or write after they get EAGAIN.  When an FD needs to wait,
// call s.WaitRead() or s.WaitWrite() to pass the request to the poll server.
// When the pollServer finds that i/o on FD should be possible
// again, it will send on fd.cr/fd.cw to wake any waiting goroutines.
//
// To avoid races in closing, all fd operations are locked and
// refcounted. when netFD.Close() is called, it calls syscall.Shutdown
// and sets a closing flag. Only when the last reference is removed
// will the fd be closed.

type pollServer struct {
	pr, pw     *os.File
	poll       *pollster // low-level OS hooks
	sync.Mutex           // controls pending and deadline
	pending    map[int]*netFD
	deadline   int64 // next deadline (nsec since 1970)
}

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

func (s *pollServer) AddFD(fd *netFD, mode int) error {
	s.Lock()
	intfd := fd.sysfd
	if intfd < 0 || fd.closing {
		// fd closed underfoot
		s.Unlock()
		return errClosing
	}

	var t int64
	key := intfd << 1
	if mode == 'r' {
		fd.ncr++
		t = fd.rdeadline.value()
	} else {
		fd.ncw++
		key++
		t = fd.wdeadline.value()
	}
	s.pending[key] = fd
	doWakeup := false
	if t > 0 && (s.deadline == 0 || t < s.deadline) {
		s.deadline = t
		doWakeup = true
	}

	wake, err := s.poll.AddFD(intfd, mode, false)
	s.Unlock()
	if err != nil {
		return &OpError{"addfd", fd.net, fd.laddr, err}
	}
	if wake || doWakeup {
		s.Wakeup()
	}
	return nil
}

// Evict evicts fd from the pending list, unblocking
// any I/O running on fd.  The caller must have locked
// pollserver.
// Return value is whether the pollServer should be woken up.
func (s *pollServer) Evict(fd *netFD) bool {
	doWakeup := false
	if s.pending[fd.sysfd<<1] == fd {
		s.WakeFD(fd, 'r', errClosing)
		if s.poll.DelFD(fd.sysfd, 'r') {
			doWakeup = true
		}
		delete(s.pending, fd.sysfd<<1)
	}
	if s.pending[fd.sysfd<<1|1] == fd {
		s.WakeFD(fd, 'w', errClosing)
		if s.poll.DelFD(fd.sysfd, 'w') {
			doWakeup = true
		}
		delete(s.pending, fd.sysfd<<1|1)
	}
	return doWakeup
}

var wakeupbuf [1]byte

func (s *pollServer) Wakeup() { s.pw.Write(wakeupbuf[0:]) }

func (s *pollServer) LookupFD(fd int, mode int) *netFD {
	key := fd << 1
	if mode == 'w' {
		key++
	}
	netfd, ok := s.pending[key]
	if !ok {
		return nil
	}
	delete(s.pending, key)
	return netfd
}

func (s *pollServer) WakeFD(fd *netFD, mode int, err error) {
	if mode == 'r' {
		for fd.ncr > 0 {
			fd.ncr--
			fd.cr <- err
		}
	} else {
		for fd.ncw > 0 {
			fd.ncw--
			fd.cw <- err
		}
	}
}

func (s *pollServer) CheckDeadlines() {
	now := time.Now().UnixNano()
	// TODO(rsc): This will need to be handled more efficiently,
	// probably with a heap indexed by wakeup time.

	var nextDeadline int64
	for key, fd := range s.pending {
		var t int64
		var mode int
		if key&1 == 0 {
			mode = 'r'
		} else {
			mode = 'w'
		}
		if mode == 'r' {
			t = fd.rdeadline.value()
		} else {
			t = fd.wdeadline.value()
		}
		if t > 0 {
			if t <= now {
				delete(s.pending, key)
				s.poll.DelFD(fd.sysfd, mode)
				s.WakeFD(fd, mode, errTimeout)
			} else if nextDeadline == 0 || t < nextDeadline {
				nextDeadline = t
			}
		}
	}
	s.deadline = nextDeadline
}

func (s *pollServer) Run() {
	var scratch [100]byte
	s.Lock()
	defer s.Unlock()
	for {
		var timeout int64 // nsec to wait for or 0 for none
		if s.deadline > 0 {
			timeout = s.deadline - time.Now().UnixNano()
			if timeout <= 0 {
				s.CheckDeadlines()
				continue
			}
		}
		fd, mode, err := s.poll.WaitFD(s, timeout)
		if err != nil {
			print("pollServer WaitFD: ", err.Error(), "\n")
			return
		}
		if fd < 0 {
			// Timeout happened.
			s.CheckDeadlines()
			continue
		}
		if fd == int(s.pr.Fd()) {
			// Drain our wakeup pipe (we could loop here,
			// but it's unlikely that there are more than
			// len(scratch) wakeup calls).
			s.pr.Read(scratch[0:])
			s.CheckDeadlines()
		} else {
			netfd := s.LookupFD(fd, mode)
			if netfd == nil {
				// This can happen because the WaitFD runs without
				// holding s's lock, so there might be a pending wakeup
				// for an fd that has been evicted.  No harm done.
				continue
			}
			s.WakeFD(netfd, mode, nil)
		}
	}
}

func (s *pollServer) PrepareRead(fd *netFD) error {
	if fd.rdeadline.expired() {
		return errTimeout
	}
	return nil
}

func (s *pollServer) PrepareWrite(fd *netFD) error {
	if fd.wdeadline.expired() {
		return errTimeout
	}
	return nil
}

func (s *pollServer) WaitRead(fd *netFD) error {
	err := s.AddFD(fd, 'r')
	if err == nil {
		err = <-fd.cr
	}
	return err
}

func (s *pollServer) WaitWrite(fd *netFD) error {
	err := s.AddFD(fd, 'w')
	if err == nil {
		err = <-fd.cw
	}
	return err
}

// Spread network FDs over several pollServers.

var pollMaxN int
var pollservers []*pollServer
var startServersOnce []func()

var canCancelIO = true // used for testing current package

func sysInit() {
	pollMaxN = runtime.NumCPU()
	if pollMaxN > 8 {
		pollMaxN = 8 // No improvement then.
	}
	pollservers = make([]*pollServer, pollMaxN)
	startServersOnce = make([]func(), pollMaxN)
	for i := 0; i < pollMaxN; i++ {
		k := i
		once := new(sync.Once)
		startServersOnce[i] = func() { once.Do(func() { startServer(k) }) }
	}
}

func startServer(k int) {
	p, err := newPollServer()
	if err != nil {
		panic(err)
	}
	pollservers[k] = p
}

func pollServerInit(fd *netFD) error {
	pollN := runtime.GOMAXPROCS(0)
	if pollN > pollMaxN {
		pollN = pollMaxN
	}
	k := fd.sysfd % pollN
	startServersOnce[k]()
	fd.pollServer = pollservers[k]
	fd.cr = make(chan error, 1)
	fd.cw = make(chan error, 1)
	return nil
}

func (s *pollServer) Close(fd *netFD) {
}

// TODO(dfc) these unused error returns could be removed

func setReadDeadline(fd *netFD, t time.Time) error {
	fd.rdeadline.setTime(t)
	return nil
}

func setWriteDeadline(fd *netFD, t time.Time) error {
	fd.wdeadline.setTime(t)
	return nil
}

func setDeadline(fd *netFD, t time.Time) error {
	setReadDeadline(fd, t)
	setWriteDeadline(fd, t)
	return nil
}

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd netbsd openbsd

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
	pending    map[int]*pollDesc
	deadline   int64 // next deadline (nsec since 1970)
}

// A pollDesc contains netFD state related to pollServer.
type pollDesc struct {
	// immutable after Init()
	pollServer *pollServer
	sysfd      int
	cr, cw     chan error

	// mutable, protected by pollServer mutex
	closing  bool
	ncr, ncw int

	// mutable, safe for concurrent access
	rdeadline, wdeadline deadline
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
	s.pending = make(map[int]*pollDesc)
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

func (s *pollServer) AddFD(pd *pollDesc, mode int) error {
	s.Lock()
	intfd := pd.sysfd
	if intfd < 0 || pd.closing {
		// fd closed underfoot
		s.Unlock()
		return errClosing
	}

	var t int64
	key := intfd << 1
	if mode == 'r' {
		pd.ncr++
		t = pd.rdeadline.value()
	} else {
		pd.ncw++
		key++
		t = pd.wdeadline.value()
	}
	s.pending[key] = pd
	doWakeup := false
	if t > 0 && (s.deadline == 0 || t < s.deadline) {
		s.deadline = t
		doWakeup = true
	}

	wake, err := s.poll.AddFD(intfd, mode, false)
	s.Unlock()
	if err != nil {
		return err
	}
	if wake || doWakeup {
		s.Wakeup()
	}
	return nil
}

// Evict evicts pd from the pending list, unblocking
// any I/O running on pd.  The caller must have locked
// pollserver.
// Return value is whether the pollServer should be woken up.
func (s *pollServer) Evict(pd *pollDesc) bool {
	pd.closing = true
	doWakeup := false
	if s.pending[pd.sysfd<<1] == pd {
		s.WakeFD(pd, 'r', errClosing)
		if s.poll.DelFD(pd.sysfd, 'r') {
			doWakeup = true
		}
		delete(s.pending, pd.sysfd<<1)
	}
	if s.pending[pd.sysfd<<1|1] == pd {
		s.WakeFD(pd, 'w', errClosing)
		if s.poll.DelFD(pd.sysfd, 'w') {
			doWakeup = true
		}
		delete(s.pending, pd.sysfd<<1|1)
	}
	return doWakeup
}

var wakeupbuf [1]byte

func (s *pollServer) Wakeup() { s.pw.Write(wakeupbuf[0:]) }

func (s *pollServer) LookupFD(fd int, mode int) *pollDesc {
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

func (s *pollServer) WakeFD(pd *pollDesc, mode int, err error) {
	if mode == 'r' {
		for pd.ncr > 0 {
			pd.ncr--
			pd.cr <- err
		}
	} else {
		for pd.ncw > 0 {
			pd.ncw--
			pd.cw <- err
		}
	}
}

func (s *pollServer) CheckDeadlines() {
	now := time.Now().UnixNano()
	// TODO(rsc): This will need to be handled more efficiently,
	// probably with a heap indexed by wakeup time.

	var nextDeadline int64
	for key, pd := range s.pending {
		var t int64
		var mode int
		if key&1 == 0 {
			mode = 'r'
		} else {
			mode = 'w'
		}
		if mode == 'r' {
			t = pd.rdeadline.value()
		} else {
			t = pd.wdeadline.value()
		}
		if t > 0 {
			if t <= now {
				delete(s.pending, key)
				s.poll.DelFD(pd.sysfd, mode)
				s.WakeFD(pd, mode, errTimeout)
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
			pd := s.LookupFD(fd, mode)
			if pd == nil {
				// This can happen because the WaitFD runs without
				// holding s's lock, so there might be a pending wakeup
				// for an fd that has been evicted.  No harm done.
				continue
			}
			s.WakeFD(pd, mode, nil)
		}
	}
}

func (pd *pollDesc) Close() {
}

func (pd *pollDesc) Lock() {
	pd.pollServer.Lock()
}

func (pd *pollDesc) Unlock() {
	pd.pollServer.Unlock()
}

func (pd *pollDesc) Wakeup() {
	pd.pollServer.Wakeup()
}

func (pd *pollDesc) PrepareRead() error {
	if pd.rdeadline.expired() {
		return errTimeout
	}
	return nil
}

func (pd *pollDesc) PrepareWrite() error {
	if pd.wdeadline.expired() {
		return errTimeout
	}
	return nil
}

func (pd *pollDesc) WaitRead() error {
	err := pd.pollServer.AddFD(pd, 'r')
	if err == nil {
		err = <-pd.cr
	}
	return err
}

func (pd *pollDesc) WaitWrite() error {
	err := pd.pollServer.AddFD(pd, 'w')
	if err == nil {
		err = <-pd.cw
	}
	return err
}

func (pd *pollDesc) Evict() bool {
	return pd.pollServer.Evict(pd)
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

func (pd *pollDesc) Init(fd *netFD) error {
	pollN := runtime.GOMAXPROCS(0)
	if pollN > pollMaxN {
		pollN = pollMaxN
	}
	k := fd.sysfd % pollN
	startServersOnce[k]()
	pd.sysfd = fd.sysfd
	pd.pollServer = pollservers[k]
	pd.cr = make(chan error, 1)
	pd.cw = make(chan error, 1)
	return nil
}

// TODO(dfc) these unused error returns could be removed

func setReadDeadline(fd *netFD, t time.Time) error {
	fd.pd.rdeadline.setTime(t)
	return nil
}

func setWriteDeadline(fd *netFD, t time.Time) error {
	fd.pd.wdeadline.setTime(t)
	return nil
}

func setDeadline(fd *netFD, t time.Time) error {
	setReadDeadline(fd, t)
	setWriteDeadline(fd, t)
	return nil
}

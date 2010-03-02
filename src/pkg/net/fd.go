// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): All the prints in this file should go to standard error.

package net

import (
	"once"
	"os"
	"sync"
	"syscall"
)

// Network file descriptor.
type netFD struct {
	// locking/lifetime of sysfd
	sysmu   sync.Mutex
	sysref  int
	closing bool

	// immutable until Close
	sysfd   int
	family  int
	proto   int
	sysfile *os.File
	cr      chan *netFD
	cw      chan *netFD
	net     string
	laddr   Addr
	raddr   Addr

	// owned by client
	rdeadline_delta int64
	rdeadline       int64
	rio             sync.Mutex
	wdeadline_delta int64
	wdeadline       int64
	wio             sync.Mutex

	// owned by fd wait server
	ncr, ncw int
}

// A pollServer helps FDs determine when to retry a non-blocking
// read or write after they get EAGAIN.  When an FD needs to wait,
// send the fd on s.cr (for a read) or s.cw (for a write) to pass the
// request to the poll server.  Then receive on fd.cr/fd.cw.
// When the pollServer finds that i/o on FD should be possible
// again, it will send fd on fd.cr/fd.cw to wake any waiting processes.
// This protocol is implemented as s.WaitRead() and s.WaitWrite().
//
// There is one subtlety: when sending on s.cr/s.cw, the
// poll server is probably in a system call, waiting for an fd
// to become ready.  It's not looking at the request channels.
// To resolve this, the poll server waits not just on the FDs it has
// been given but also its own pipe.  After sending on the
// buffered channel s.cr/s.cw, WaitRead/WaitWrite writes a
// byte to the pipe, causing the pollServer's poll system call to
// return.  In response to the pipe being readable, the pollServer
// re-polls its request channels.
//
// Note that the ordering is "send request" and then "wake up server".
// If the operations were reversed, there would be a race: the poll
// server might wake up and look at the request channel, see that it
// was empty, and go back to sleep, all before the requester managed
// to send the request.  Because the send must complete before the wakeup,
// the request channel must be buffered.  A buffer of size 1 is sufficient
// for any request load.  If many processes are trying to submit requests,
// one will succeed, the pollServer will read the request, and then the
// channel will be empty for the next process's request.  A larger buffer
// might help batch requests.
//
// To avoid races in closing, all fd operations are locked and
// refcounted. when netFD.Close() is called, it calls syscall.Shutdown
// and sets a closing flag. Only when the last reference is removed
// will the fd be closed.

type pollServer struct {
	cr, cw   chan *netFD // buffered >= 1
	pr, pw   *os.File
	pending  map[int]*netFD
	poll     *pollster // low-level OS hooks
	deadline int64     // next deadline (nsec since 1970)
}

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

func (s *pollServer) AddFD(fd *netFD, mode int) {
	intfd := fd.sysfd
	if intfd < 0 {
		// fd closed underfoot
		if mode == 'r' {
			fd.cr <- fd
		} else {
			fd.cw <- fd
		}
		return
	}
	if err := s.poll.AddFD(intfd, mode, false); err != nil {
		panicln("pollServer AddFD ", intfd, ": ", err.String(), "\n")
		return
	}

	var t int64
	key := intfd << 1
	if mode == 'r' {
		fd.ncr++
		t = fd.rdeadline
	} else {
		fd.ncw++
		key++
		t = fd.wdeadline
	}
	s.pending[key] = fd
	if t > 0 && (s.deadline == 0 || t < s.deadline) {
		s.deadline = t
	}
}

func (s *pollServer) LookupFD(fd int, mode int) *netFD {
	key := fd << 1
	if mode == 'w' {
		key++
	}
	netfd, ok := s.pending[key]
	if !ok {
		return nil
	}
	s.pending[key] = nil, false
	return netfd
}

func (s *pollServer) WakeFD(fd *netFD, mode int) {
	if mode == 'r' {
		for fd.ncr > 0 {
			fd.ncr--
			fd.cr <- fd
		}
	} else {
		for fd.ncw > 0 {
			fd.ncw--
			fd.cw <- fd
		}
	}
}

func (s *pollServer) Now() int64 {
	sec, nsec, err := os.Time()
	if err != nil {
		panic("net: os.Time: ", err.String())
	}
	nsec += sec * 1e9
	return nsec
}

func (s *pollServer) CheckDeadlines() {
	now := s.Now()
	// TODO(rsc): This will need to be handled more efficiently,
	// probably with a heap indexed by wakeup time.

	var next_deadline int64
	for key, fd := range s.pending {
		var t int64
		var mode int
		if key&1 == 0 {
			mode = 'r'
		} else {
			mode = 'w'
		}
		if mode == 'r' {
			t = fd.rdeadline
		} else {
			t = fd.wdeadline
		}
		if t > 0 {
			if t <= now {
				s.pending[key] = nil, false
				if mode == 'r' {
					s.poll.DelFD(fd.sysfd, mode)
					fd.rdeadline = -1
				} else {
					s.poll.DelFD(fd.sysfd, mode)
					fd.wdeadline = -1
				}
				s.WakeFD(fd, mode)
			} else if next_deadline == 0 || t < next_deadline {
				next_deadline = t
			}
		}
	}
	s.deadline = next_deadline
}

func (s *pollServer) Run() {
	var scratch [100]byte
	for {
		var t = s.deadline
		if t > 0 {
			t = t - s.Now()
			if t < 0 {
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
			for nn, _ := s.pr.Read(&scratch); nn > 0; {
				nn, _ = s.pr.Read(&scratch)
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

var wakeupbuf [1]byte

func (s *pollServer) Wakeup() { s.pw.Write(&wakeupbuf) }

func (s *pollServer) WaitRead(fd *netFD) {
	s.cr <- fd
	s.Wakeup()
	<-fd.cr
}

func (s *pollServer) WaitWrite(fd *netFD) {
	s.cw <- fd
	s.Wakeup()
	<-fd.cw
}

// Network FD methods.
// All the network FDs use a single pollServer.

var pollserver *pollServer

func startServer() {
	p, err := newPollServer()
	if err != nil {
		print("Start pollServer: ", err.String(), "\n")
	}
	pollserver = p
}

func newFD(fd, family, proto int, net string, laddr, raddr Addr) (f *netFD, err os.Error) {
	once.Do(startServer)
	if e := syscall.SetNonblock(fd, true); e != 0 {
		return nil, &OpError{"setnonblock", net, laddr, os.Errno(e)}
	}
	f = &netFD{
		sysfd:  fd,
		family: family,
		proto:  proto,
		net:    net,
		laddr:  laddr,
		raddr:  raddr,
	}
	var ls, rs string
	if laddr != nil {
		ls = laddr.String()
	}
	if raddr != nil {
		rs = raddr.String()
	}
	f.sysfile = os.NewFile(fd, net+":"+ls+"->"+rs)
	f.cr = make(chan *netFD, 1)
	f.cw = make(chan *netFD, 1)
	return f, nil
}

// Add a reference to this fd.
func (fd *netFD) incref() {
	fd.sysmu.Lock()
	fd.sysref++
	fd.sysmu.Unlock()
}

// Remove a reference to this FD and close if we've been asked to do so (and
// there are no references left.
func (fd *netFD) decref() {
	fd.sysmu.Lock()
	fd.sysref--
	if fd.closing && fd.sysref == 0 && fd.sysfd >= 0 {
		// In case the user has set linger, switch to blocking mode so
		// the close blocks.  As long as this doesn't happen often, we
		// can handle the extra OS processes.  Otherwise we'll need to
		// use the pollserver for Close too.  Sigh.
		syscall.SetNonblock(fd.sysfd, false)
		fd.sysfile.Close()
		fd.sysfile = nil
		fd.sysfd = -1
	}
	fd.sysmu.Unlock()
}

func isEAGAIN(e os.Error) bool {
	if e1, ok := e.(*os.PathError); ok {
		return e1.Error == os.EAGAIN
	}
	return e == os.EAGAIN
}

func (fd *netFD) Close() os.Error {
	if fd == nil || fd.sysfile == nil {
		return os.EINVAL
	}

	fd.incref()
	syscall.Shutdown(fd.sysfd, syscall.SHUT_RDWR)
	fd.closing = true
	fd.decref()
	return nil
}

func (fd *netFD) Read(p []byte) (n int, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return 0, os.EINVAL
	}
	fd.rio.Lock()
	defer fd.rio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.rdeadline_delta > 0 {
		fd.rdeadline = pollserver.Now() + fd.rdeadline_delta
	} else {
		fd.rdeadline = 0
	}
	for {
		n, err = fd.sysfile.Read(p)
		if isEAGAIN(err) && fd.rdeadline >= 0 {
			pollserver.WaitRead(fd)
			continue
		}
		break
	}
	if fd.proto == syscall.SOCK_DGRAM && err == os.EOF {
		// 0 in datagram protocol just means 0-length packet
		err = nil
	}
	return
}

func (fd *netFD) ReadFrom(p []byte) (n int, sa syscall.Sockaddr, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return 0, nil, os.EINVAL
	}
	fd.rio.Lock()
	defer fd.rio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.rdeadline_delta > 0 {
		fd.rdeadline = pollserver.Now() + fd.rdeadline_delta
	} else {
		fd.rdeadline = 0
	}
	for {
		var errno int
		n, sa, errno = syscall.Recvfrom(fd.sysfd, p, 0)
		if errno == syscall.EAGAIN && fd.rdeadline >= 0 {
			pollserver.WaitRead(fd)
			continue
		}
		if errno != 0 {
			n = 0
			err = &os.PathError{"recvfrom", fd.sysfile.Name(), os.Errno(errno)}
		}
		break
	}
	return
}

func (fd *netFD) Write(p []byte) (n int, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return 0, os.EINVAL
	}
	fd.wio.Lock()
	defer fd.wio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.wdeadline_delta > 0 {
		fd.wdeadline = pollserver.Now() + fd.wdeadline_delta
	} else {
		fd.wdeadline = 0
	}
	err = nil
	nn := 0
	first := true // force at least one Write, to send 0-length datagram packets
	for nn < len(p) || first {
		first = false
		n, err = fd.sysfile.Write(p[nn:])
		if n > 0 {
			nn += n
		}
		if nn == len(p) {
			break
		}
		if isEAGAIN(err) && fd.wdeadline >= 0 {
			pollserver.WaitWrite(fd)
			continue
		}
		if n == 0 || err != nil {
			break
		}
	}
	return nn, err
}

func (fd *netFD) WriteTo(p []byte, sa syscall.Sockaddr) (n int, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return 0, os.EINVAL
	}
	fd.wio.Lock()
	defer fd.wio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.wdeadline_delta > 0 {
		fd.wdeadline = pollserver.Now() + fd.wdeadline_delta
	} else {
		fd.wdeadline = 0
	}
	err = nil
	for {
		errno := syscall.Sendto(fd.sysfd, p, 0, sa)
		if errno == syscall.EAGAIN && fd.wdeadline >= 0 {
			pollserver.WaitWrite(fd)
			continue
		}
		if errno != 0 {
			err = &os.PathError{"sendto", fd.sysfile.Name(), os.Errno(errno)}
		}
		break
	}
	if err == nil {
		n = len(p)
	}
	return
}

func (fd *netFD) accept(toAddr func(syscall.Sockaddr) Addr) (nfd *netFD, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return nil, os.EINVAL
	}

	fd.incref()
	defer fd.decref()

	// See ../syscall/exec.go for description of ForkLock.
	// It is okay to hold the lock across syscall.Accept
	// because we have put fd.sysfd into non-blocking mode.
	syscall.ForkLock.RLock()
	var s, e int
	var sa syscall.Sockaddr
	for {
		s, sa, e = syscall.Accept(fd.sysfd)
		if e != syscall.EAGAIN {
			break
		}
		syscall.ForkLock.RUnlock()
		pollserver.WaitRead(fd)
		syscall.ForkLock.RLock()
	}
	if e != 0 {
		syscall.ForkLock.RUnlock()
		return nil, &OpError{"accept", fd.net, fd.laddr, os.Errno(e)}
	}
	syscall.CloseOnExec(s)
	syscall.ForkLock.RUnlock()

	if nfd, err = newFD(s, fd.family, fd.proto, fd.net, fd.laddr, toAddr(sa)); err != nil {
		syscall.Close(s)
		return nil, err
	}
	return nfd, nil
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): All the prints in this file should go to standard error.

package net

import (
	"net";
	"once";
	"os";
	"sync";
	"syscall";
)

// Network file descriptor.
type netFD struct {
	// immutable until Close
	fd int64;
	file *os.File;
	cr chan *netFD;
	cw chan *netFD;
	net string;
	laddr string;
	raddr string;

	// owned by client
	rdeadline_delta int64;
	rdeadline int64;
	rio sync.Mutex;
	wdeadline_delta int64;
	wdeadline int64;
	wio sync.Mutex;

	// owned by fd wait server
	ncr, ncw int;
}

// Make reads and writes on fd return EAGAIN instead of blocking.
func setNonblock(fd int64) os.Error {
	flags, e := syscall.Fcntl(fd, syscall.F_GETFL, 0);
	if e != 0 {
		return os.ErrnoToError(e)
	}
	flags, e = syscall.Fcntl(fd, syscall.F_SETFL, flags | syscall.O_NONBLOCK);
	if e != 0 {
		return os.ErrnoToError(e)
	}
	return nil
}

// Make reads/writes blocking; last gasp, so no error checking.
func setBlock(fd int64) {
	flags, e := syscall.Fcntl(fd, syscall.F_GETFL, 0);
	if e != 0 {
		return;
	}
	syscall.Fcntl(fd, syscall.F_SETFL, flags & ^syscall.O_NONBLOCK);
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

type pollServer struct {
	cr, cw chan *netFD;	// buffered >= 1
	pr, pw *os.File;
	pending map[int64] *netFD;
	poll *pollster;	// low-level OS hooks
	deadline int64;	// next deadline (nsec since 1970)
}
func (s *pollServer) Run();

func newPollServer() (s *pollServer, err os.Error) {
	s = new(pollServer);
	s.cr = make(chan *netFD, 1);
	s.cw = make(chan *netFD, 1);
	if s.pr, s.pw, err = os.Pipe(); err != nil {
		return nil, err
	}
	if err = setNonblock(s.pr.Fd()); err != nil {
	Error:
		s.pr.Close();
		s.pw.Close();
		return nil, err
	}
	if err = setNonblock(s.pw.Fd()); err != nil {
		goto Error
	}
	if s.poll, err = newpollster(); err != nil {
		goto Error
	}
	if err = s.poll.AddFD(s.pr.Fd(), 'r', true); err != nil {
		s.poll.Close();
		goto Error
	}
	s.pending = make(map[int64] *netFD);
	go s.Run();
	return s, nil
}

func (s *pollServer) AddFD(fd *netFD, mode int) {
	// TODO(rsc): This check handles a race between
	// one goroutine reading and another one closing,
	// but it doesn't solve the race completely:
	// it still could happen that one goroutine closes
	// but we read fd.fd before it does, and then
	// another goroutine creates a new open file with
	// that fd, which we'd now be referring to.
	// The fix is probably to send the Close call
	// through the poll server too, except that
	// not all Reads and Writes go through the poll
	// server even now.
	intfd := fd.fd;
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
		panicln("pollServer AddFD ", intfd, ": ", err.String(), "\n");
		return
	}

	var t int64;
	key := intfd << 1;
	if mode == 'r' {
		fd.ncr++;
		t = fd.rdeadline;
	} else {
		fd.ncw++;
		key++;
		t = fd.wdeadline;
	}
	s.pending[key] = fd;
	if t > 0 && (s.deadline == 0 || t < s.deadline) {
		s.deadline = t;
	}
}

func (s *pollServer) LookupFD(fd int64, mode int) *netFD {
	key := fd << 1;
	if mode == 'w' {
		key++;
	}
	netfd, ok := s.pending[key];
	if !ok {
		return nil
	}
	s.pending[key] = nil, false;
	return netfd
}

func (s *pollServer) WakeFD(fd *netFD, mode int) {
	if mode == 'r' {
		for fd.ncr > 0 {
			fd.ncr--;
			fd.cr <- fd
		}
	} else {
		for fd.ncw > 0 {
			fd.ncw--;
			fd.cw <- fd
		}
	}
}

func (s *pollServer) Now() int64 {
	sec, nsec, err := os.Time();
	if err != nil {
		panic("net: os.Time: ", err.String());
	}
	nsec += sec * 1e9;
	return nsec;
}

func (s *pollServer) CheckDeadlines() {
	now := s.Now();
	// TODO(rsc): This will need to be handled more efficiently,
	// probably with a heap indexed by wakeup time.

	var next_deadline int64;
	for key, fd := range s.pending {
		var t int64;
		var mode int;
		if key&1 == 0 {
			mode = 'r';
		} else {
			mode = 'w';
		}
		if mode == 'r' {
			t = fd.rdeadline;
		} else {
			t = fd.wdeadline;
		}
		if t > 0 {
			if t <= now {
				s.pending[key] = nil, false;
				if mode == 'r' {
					s.poll.DelFD(fd.fd, mode);
					fd.rdeadline = -1;
				} else {
					s.poll.DelFD(fd.fd, mode);
					fd.wdeadline = -1;
				}
				s.WakeFD(fd, mode);
			} else if next_deadline == 0 || t < next_deadline {
				next_deadline = t;
			}
		}
	}
	s.deadline = next_deadline;
}

func (s *pollServer) Run() {
	var scratch [100]byte;
	for {
		var t = s.deadline;
		if t > 0 {
			t = t - s.Now();
			if t < 0 {
				s.CheckDeadlines();
				continue;
			}
		}
		fd, mode, err := s.poll.WaitFD(t);
		if err != nil {
			print("pollServer WaitFD: ", err.String(), "\n");
			return
		}
		if fd < 0 {
			// Timeout happened.
			s.CheckDeadlines();
			continue;
		}
		if fd == s.pr.Fd() {
			// Drain our wakeup pipe.
			for nn, e := s.pr.Read(&scratch); nn > 0; {
				nn, e = s.pr.Read(&scratch)
			}

			// Read from channels
			for fd, ok := <-s.cr; ok; fd, ok = <-s.cr {
				s.AddFD(fd, 'r')
			}
			for fd, ok := <-s.cw; ok; fd, ok = <-s.cw {
				s.AddFD(fd, 'w')
			}
		} else {
			netfd := s.LookupFD(fd, mode);
			if netfd == nil {
				print("pollServer: unexpected wakeup for fd=", netfd, " mode=", string(mode), "\n");
				continue
			}
			s.WakeFD(netfd, mode);
		}
	}
}

var wakeupbuf [1]byte;
func (s *pollServer) Wakeup() {
	s.pw.Write(&wakeupbuf)
}

func (s *pollServer) WaitRead(fd *netFD) {
	s.cr <- fd;
	s.Wakeup();
	<-fd.cr
}

func (s *pollServer) WaitWrite(fd *netFD) {
	s.cr <- fd;
	s.Wakeup();
	<-fd.cr
}


// Network FD methods.
// All the network FDs use a single pollServer.

var pollserver *pollServer

func _StartServer() {
	p, err := newPollServer();
	if err != nil {
		print("Start pollServer: ", err.String(), "\n")
	}
	pollserver = p
}

func newFD(fd int64, net, laddr, raddr string) (f *netFD, err os.Error) {
	if pollserver == nil {
		once.Do(_StartServer);
	}
	if err = setNonblock(fd); err != nil {
		return nil, err
	}
	f = new(netFD);
	f.fd = fd;
	f.net = net;
	f.laddr = laddr;
	f.raddr = raddr;
	f.file = os.NewFile(fd, "net: " + net + " " + laddr + " " + raddr);
	f.cr = make(chan *netFD, 1);
	f.cw = make(chan *netFD, 1);
	return f, nil
}

func (fd *netFD) Close() os.Error {
	if fd == nil || fd.file == nil {
		return os.EINVAL
	}

	// In case the user has set linger,
	// switch to blocking mode so the close blocks.
	// As long as this doesn't happen often,
	// we can handle the extra OS processes.
	// Otherwise we'll need to use the pollserver
	// for Close too.  Sigh.
	setBlock(fd.file.Fd());

	e := fd.file.Close();
	fd.file = nil;
	fd.fd = -1;
	return e
}

func (fd *netFD) Read(p []byte) (n int, err os.Error) {
	if fd == nil || fd.file == nil {
		return -1, os.EINVAL
	}
	fd.rio.Lock();
	defer fd.rio.Unlock();
	if fd.rdeadline_delta > 0 {
		fd.rdeadline = pollserver.Now() + fd.rdeadline_delta;
	} else {
		fd.rdeadline = 0;
	}
	n, err = fd.file.Read(p);
	for err == os.EAGAIN && fd.rdeadline >= 0 {
		pollserver.WaitRead(fd);
		n, err = fd.file.Read(p)
	}
	return n, err
}

func (fd *netFD) Write(p []byte) (n int, err os.Error) {
	if fd == nil || fd.file == nil {
		return -1, os.EINVAL
	}
	fd.wio.Lock();
	defer fd.wio.Unlock();
	if fd.wdeadline_delta > 0 {
		fd.wdeadline = pollserver.Now() + fd.wdeadline_delta;
	} else {
		fd.wdeadline = 0;
	}
	err = nil;
	nn := 0;
	for nn < len(p) {
		n, err = fd.file.Write(p[nn:len(p)]);
		if n > 0 {
			nn += n
		}
		if nn == len(p) {
			break;
		}
		if err == os.EAGAIN && fd.wdeadline >= 0 {
			pollserver.WaitWrite(fd);
			continue;
		}
		if n == 0 || err != nil {
			break;
		}
	}
	return nn, err
}

func sockaddrToHostPort(sa *syscall.Sockaddr) (hostport string, err os.Error)

func (fd *netFD) Accept(sa *syscall.Sockaddr) (nfd *netFD, err os.Error) {
	if fd == nil || fd.file == nil {
		return nil, os.EINVAL
	}

	// See ../syscall/exec.go for description of ForkLock.
	// It is okay to hold the lock across syscall.Accept
	// because we have put fd.fd into non-blocking mode.
	syscall.ForkLock.RLock();
	var s, e int64;
	for {
		s, e = syscall.Accept(fd.fd, sa);
		if e != syscall.EAGAIN {
			break;
		}
		syscall.ForkLock.RUnlock();
		pollserver.WaitRead(fd);
		syscall.ForkLock.RLock();
	}
	if e != 0 {
		syscall.ForkLock.RUnlock();
		return nil, os.ErrnoToError(e)
	}
	syscall.CloseOnExec(s);
	syscall.ForkLock.RUnlock();

	raddr, err1 := sockaddrToHostPort(sa);
	if err1 != nil {
		raddr = "invalid-address";
	}
	if nfd, err = newFD(s, fd.net, fd.laddr, raddr); err != nil {
		syscall.Close(s);
		return nil, err
	}
	return nfd, nil
}


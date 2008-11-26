// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): All the prints in this file should go to standard error.

package net

import (
	"net";
	"once";
	"os";
	"syscall";
)

// Network file descriptor.  Only intended to be used internally,
// but have to export to make it available in other files implementing package net.
export type FD struct {
	// immutable until Close
	fd int64;
	osfd *os.FD;
	cr *chan *FD;
	cw *chan *FD;

	// owned by fd wait server
	ncr, ncw int;
}

// Make reads and writes on fd return EAGAIN instead of blocking.
func SetNonblock(fd int64) *os.Error {
	flags, e := syscall.fcntl(fd, syscall.F_GETFL, 0);
	if e != 0 {
		return os.ErrnoToError(e)
	}
	flags, e = syscall.fcntl(fd, syscall.F_SETFL, flags | syscall.O_NONBLOCK);
	if e != 0 {
		return os.ErrnoToError(e)
	}
	return nil
}


// A PollServer helps FDs determine when to retry a non-blocking
// read or write after they get EAGAIN.  When an FD needs to wait,
// send the fd on s.cr (for a read) or s.cw (for a write) to pass the
// request to the poll server.  Then receive on fd.cr/fd.cw.
// When the PollServer finds that i/o on FD should be possible
// again, it will send fd on fd.cr/fd.cw to wake any waiting processes.
// This protocol is implemented as s.WaitRead() and s.WaitWrite().
//
// There is one subtlety: when sending on s.cr/s.cw, the
// poll server is probably in a system call, waiting for an fd
// to become ready.  It's not looking at the request channels.
// To resolve this, the poll server waits not just on the FDs it has
// been given but also its own pipe.  After sending on the
// buffered channel s.cr/s.cw, WaitRead/WaitWrite writes a
// byte to the pipe, causing the PollServer's poll system call to
// return.  In response to the pipe being readable, the PollServer
// re-polls its request channels.
//
// Note that the ordering is "send request" and then "wake up server".
// If the operations were reversed, there would be a race: the poll
// server might wake up and look at the request channel, see that it
// was empty, and go back to sleep, all before the requester managed
// to send the request.  Because the send must complete before the wakeup,
// the request channel must be buffered.  A buffer of size 1 is sufficient
// for any request load.  If many processes are trying to submit requests,
// one will succeed, the PollServer will read the request, and then the
// channel will be empty for the next process's request.  A larger buffer
// might help batch requests.

type PollServer struct {
	cr, cw *chan *FD;	// buffered >= 1
	pr, pw *os.FD;
	pending *map[int64] *FD;
	poll *Pollster;	// low-level OS hooks
}
func (s *PollServer) Run();

func NewPollServer() (s *PollServer, err *os.Error) {
	s = new(PollServer);
	s.cr = new(chan *FD, 1);
	s.cw = new(chan *FD, 1);
	if s.pr, s.pw, err = os.Pipe(); err != nil {
		return nil, err
	}
	if err = SetNonblock(s.pr.fd); err != nil {
	Error:
		s.pr.Close();
		s.pw.Close();
		return nil, err
	}
	if err = SetNonblock(s.pw.fd); err != nil {
		goto Error
	}
	if s.poll, err = NewPollster(); err != nil {
		goto Error
	}
	if err = s.poll.AddFD(s.pr.fd, 'r', true); err != nil {
		s.poll.Close();
		goto Error
	}
	s.pending = new(map[int64] *FD);
	go s.Run();
	return s, nil
}

func (s *PollServer) AddFD(fd *FD, mode int) {
	if err := s.poll.AddFD(fd.fd, mode, false); err != nil {
		print("PollServer AddFD: ", err.String(), "\n");
		return
	}

	key := fd.fd << 1;
	if mode == 'r' {
		fd.ncr++;
	} else {
		fd.ncw++;
		key++;
	}
	s.pending[key] = fd
}

func (s *PollServer) LookupFD(fd int64, mode int) *FD {
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

func (s *PollServer) Run() {
	var scratch [100]byte;
	for {
		fd, mode, err := s.poll.WaitFD();
		if err != nil {
			print("PollServer WaitFD: ", err.String(), "\n");
			return
		}
		if fd == s.pr.fd {
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
				print("PollServer: unexpected wakeup for fd=", netfd, " mode=", string(mode), "\n");
				continue
			}
			if mode == 'r' {
				for netfd.ncr > 0 {
					netfd.ncr--;
					netfd.cr <- netfd
				}
			} else {
				for netfd.ncw > 0 {
					netfd.ncw--;
					netfd.cw <- netfd
				}
			}
		}
	}
}

func (s *PollServer) Wakeup() {
	var b [1]byte;
	s.pw.Write(&b)
}

func (s *PollServer) WaitRead(fd *FD) {
	s.cr <- fd;
	s.Wakeup();
	<-fd.cr
}

func (s *PollServer) WaitWrite(fd *FD) {
	s.cr <- fd;
	s.Wakeup();
	<-fd.cr
}


// Network FD methods.
// All the network FDs use a single PollServer.

var pollserver *PollServer

func StartServer() {
	p, err := NewPollServer();
	if err != nil {
		print("Start PollServer: ", err.String(), "\n")
	}
	pollserver = p
}

export func NewFD(fd int64) (f *FD, err *os.Error) {
	if pollserver == nil {
		once.Do(&StartServer);
	}
	if err = SetNonblock(fd); err != nil {
		return nil, err
	}
	f = new(FD);
	f.fd = fd;
	f.osfd = os.NewFD(fd);
	f.cr = new(chan *FD, 1);
	f.cw = new(chan *FD, 1);
	return f, nil
}

func (fd *FD) Close() *os.Error {
	if fd == nil || fd.osfd == nil {
		return os.EINVAL
	}
	e := fd.osfd.Close();
	fd.osfd = nil;
	fd.fd = -1;
	return e
}

func (fd *FD) Read(p *[]byte) (n int, err *os.Error) {
	if fd == nil || fd.osfd == nil {
		return -1, os.EINVAL
	}
	n, err = fd.osfd.Read(p);
	for err == os.EAGAIN {
		pollserver.WaitRead(fd);
		n, err = fd.osfd.Read(p)
	}
	return n, err
}

func (fd *FD) Write(p *[]byte) (n int, err *os.Error) {
	if fd == nil || fd.osfd == nil {
		return -1, os.EINVAL
	}
	// TODO(rsc): Lock fd while writing to avoid interlacing writes.
	err = nil;
	nn := 0;
	for nn < len(p) && err == nil {
		// TODO(rsc): If os.FD.Write loops, have to use syscall instead.
		n, err = fd.osfd.Write(p[nn:len(p)]);
		for err == os.EAGAIN {
			pollserver.WaitWrite(fd);
			n, err = fd.osfd.Write(p[nn:len(p)])
		}
		if n > 0 {
			nn += n
		}
		if n == 0 {
			break
		}
	}
	return nn, err
}

func (fd *FD) Accept(sa *syscall.Sockaddr) (nfd *FD, err *os.Error) {
	if fd == nil || fd.osfd == nil {
		return nil, os.EINVAL
	}
	s, e := syscall.accept(fd.fd, sa);
	for e == syscall.EAGAIN {
		pollserver.WaitRead(fd);
		s, e = syscall.accept(fd.fd, sa)
	}
	if e != 0 {
		return nil, os.ErrnoToError(e)
	}
	if nfd, err = NewFD(s); err != nil {
		syscall.close(s);
		return nil, err
	}
	return nfd, nil
}


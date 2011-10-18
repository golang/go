// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

package net

import (
	"io"
	"os"
	"sync"
	"syscall"
	"time"
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
	cr      chan bool
	cw      chan bool
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

type InvalidConnError struct{}

func (e *InvalidConnError) String() string  { return "invalid net.Conn" }
func (e *InvalidConnError) Temporary() bool { return false }
func (e *InvalidConnError) Timeout() bool   { return false }

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
	cr, cw     chan *netFD // buffered >= 1
	pr, pw     *os.File
	poll       *pollster // low-level OS hooks
	sync.Mutex           // controls pending and deadline
	pending    map[int]*netFD
	deadline   int64 // next deadline (nsec since 1970)
}

func (s *pollServer) AddFD(fd *netFD, mode int) {
	intfd := fd.sysfd
	if intfd < 0 {
		// fd closed underfoot
		if mode == 'r' {
			fd.cr <- true
		} else {
			fd.cw <- true
		}
		return
	}

	s.Lock()

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
	doWakeup := false
	if t > 0 && (s.deadline == 0 || t < s.deadline) {
		s.deadline = t
		doWakeup = true
	}

	wake, err := s.poll.AddFD(intfd, mode, false)
	if err != nil {
		panic("pollServer AddFD " + err.String())
	}
	if wake {
		doWakeup = true
	}

	s.Unlock()

	if doWakeup {
		s.Wakeup()
	}
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

func (s *pollServer) WakeFD(fd *netFD, mode int) {
	if mode == 'r' {
		for fd.ncr > 0 {
			fd.ncr--
			fd.cr <- true
		}
	} else {
		for fd.ncw > 0 {
			fd.ncw--
			fd.cw <- true
		}
	}
}

func (s *pollServer) Now() int64 {
	return time.Nanoseconds()
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
				delete(s.pending, key)
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
	s.Lock()
	defer s.Unlock()
	for {
		var t = s.deadline
		if t > 0 {
			t = t - s.Now()
			if t <= 0 {
				s.CheckDeadlines()
				continue
			}
		}
		fd, mode, err := s.poll.WaitFD(s, t)
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
			// Drain our wakeup pipe (we could loop here,
			// but it's unlikely that there are more than
			// len(scratch) wakeup calls).
			s.pr.Read(scratch[0:])
			s.CheckDeadlines()
		} else {
			netfd := s.LookupFD(fd, mode)
			if netfd == nil {
				print("pollServer: unexpected wakeup for fd=", fd, " mode=", string(mode), "\n")
				continue
			}
			s.WakeFD(netfd, mode)
		}
	}
}

func (s *pollServer) WaitRead(fd *netFD) {
	s.AddFD(fd, 'r')
	<-fd.cr
}

func (s *pollServer) WaitWrite(fd *netFD) {
	s.AddFD(fd, 'w')
	<-fd.cw
}

// Network FD methods.
// All the network FDs use a single pollServer.

var pollserver *pollServer
var onceStartServer sync.Once

func startServer() {
	p, err := newPollServer()
	if err != nil {
		print("Start pollServer: ", err.String(), "\n")
	}
	pollserver = p
}

func newFD(fd, family, proto int, net string) (f *netFD, err os.Error) {
	onceStartServer.Do(startServer)
	if e := syscall.SetNonblock(fd, true); e != 0 {
		return nil, os.Errno(e)
	}
	f = &netFD{
		sysfd:  fd,
		family: family,
		proto:  proto,
		net:    net,
	}
	f.cr = make(chan bool, 1)
	f.cw = make(chan bool, 1)
	return f, nil
}

func (fd *netFD) setAddr(laddr, raddr Addr) {
	fd.laddr = laddr
	fd.raddr = raddr
	var ls, rs string
	if laddr != nil {
		ls = laddr.String()
	}
	if raddr != nil {
		rs = raddr.String()
	}
	fd.sysfile = os.NewFile(fd.sysfd, fd.net+":"+ls+"->"+rs)
}

func (fd *netFD) connect(ra syscall.Sockaddr) (err os.Error) {
	e := syscall.Connect(fd.sysfd, ra)
	if e == syscall.EINPROGRESS {
		var errno int
		pollserver.WaitWrite(fd)
		e, errno = syscall.GetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_ERROR)
		if errno != 0 {
			return os.NewSyscallError("getsockopt", errno)
		}
	}
	if e != 0 {
		return os.Errno(e)
	}
	return nil
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

func (fd *netFD) shutdown(how int) os.Error {
	if fd == nil || fd.sysfile == nil {
		return os.EINVAL
	}
	errno := syscall.Shutdown(fd.sysfd, how)
	if errno != 0 {
		return &OpError{"shutdown", fd.net, fd.laddr, os.Errno(errno)}
	}
	return nil
}

func (fd *netFD) CloseRead() os.Error {
	return fd.shutdown(syscall.SHUT_RD)
}

func (fd *netFD) CloseWrite() os.Error {
	return fd.shutdown(syscall.SHUT_WR)
}

func (fd *netFD) Read(p []byte) (n int, err os.Error) {
	if fd == nil {
		return 0, os.EINVAL
	}
	fd.rio.Lock()
	defer fd.rio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.sysfile == nil {
		return 0, os.EINVAL
	}
	if fd.rdeadline_delta > 0 {
		fd.rdeadline = pollserver.Now() + fd.rdeadline_delta
	} else {
		fd.rdeadline = 0
	}
	var oserr os.Error
	for {
		var errno int
		n, errno = syscall.Read(fd.sysfile.Fd(), p)
		if errno == syscall.EAGAIN && fd.rdeadline >= 0 {
			pollserver.WaitRead(fd)
			continue
		}
		if errno != 0 {
			n = 0
			oserr = os.Errno(errno)
		} else if n == 0 && errno == 0 && fd.proto != syscall.SOCK_DGRAM {
			err = os.EOF
		}
		break
	}
	if oserr != nil {
		err = &OpError{"read", fd.net, fd.raddr, oserr}
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
	var oserr os.Error
	for {
		var errno int
		n, sa, errno = syscall.Recvfrom(fd.sysfd, p, 0)
		if errno == syscall.EAGAIN && fd.rdeadline >= 0 {
			pollserver.WaitRead(fd)
			continue
		}
		if errno != 0 {
			n = 0
			oserr = os.Errno(errno)
		}
		break
	}
	if oserr != nil {
		err = &OpError{"read", fd.net, fd.laddr, oserr}
	}
	return
}

func (fd *netFD) ReadMsg(p []byte, oob []byte) (n, oobn, flags int, sa syscall.Sockaddr, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return 0, 0, 0, nil, os.EINVAL
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
	var oserr os.Error
	for {
		var errno int
		n, oobn, flags, sa, errno = syscall.Recvmsg(fd.sysfd, p, oob, 0)
		if errno == syscall.EAGAIN && fd.rdeadline >= 0 {
			pollserver.WaitRead(fd)
			continue
		}
		if errno != 0 {
			oserr = os.Errno(errno)
		}
		if n == 0 {
			oserr = os.EOF
		}
		break
	}
	if oserr != nil {
		err = &OpError{"read", fd.net, fd.laddr, oserr}
		return
	}
	return
}

func (fd *netFD) Write(p []byte) (n int, err os.Error) {
	if fd == nil {
		return 0, os.EINVAL
	}
	fd.wio.Lock()
	defer fd.wio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.sysfile == nil {
		return 0, os.EINVAL
	}
	if fd.wdeadline_delta > 0 {
		fd.wdeadline = pollserver.Now() + fd.wdeadline_delta
	} else {
		fd.wdeadline = 0
	}
	nn := 0
	var oserr os.Error

	for {
		n, errno := syscall.Write(fd.sysfile.Fd(), p[nn:])
		if n > 0 {
			nn += n
		}
		if nn == len(p) {
			break
		}
		if errno == syscall.EAGAIN && fd.wdeadline >= 0 {
			pollserver.WaitWrite(fd)
			continue
		}
		if errno != 0 {
			n = 0
			oserr = os.Errno(errno)
			break
		}
		if n == 0 {
			oserr = io.ErrUnexpectedEOF
			break
		}
	}
	if oserr != nil {
		err = &OpError{"write", fd.net, fd.raddr, oserr}
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
	var oserr os.Error
	for {
		errno := syscall.Sendto(fd.sysfd, p, 0, sa)
		if errno == syscall.EAGAIN && fd.wdeadline >= 0 {
			pollserver.WaitWrite(fd)
			continue
		}
		if errno != 0 {
			oserr = os.Errno(errno)
		}
		break
	}
	if oserr == nil {
		n = len(p)
	} else {
		err = &OpError{"write", fd.net, fd.raddr, oserr}
	}
	return
}

func (fd *netFD) WriteMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return 0, 0, os.EINVAL
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
	var oserr os.Error
	for {
		var errno int
		errno = syscall.Sendmsg(fd.sysfd, p, oob, sa, 0)
		if errno == syscall.EAGAIN && fd.wdeadline >= 0 {
			pollserver.WaitWrite(fd)
			continue
		}
		if errno != 0 {
			oserr = os.Errno(errno)
		}
		break
	}
	if oserr == nil {
		n = len(p)
		oobn = len(oob)
	} else {
		err = &OpError{"write", fd.net, fd.raddr, oserr}
	}
	return
}

func (fd *netFD) accept(toAddr func(syscall.Sockaddr) Addr) (nfd *netFD, err os.Error) {
	if fd == nil || fd.sysfile == nil {
		return nil, os.EINVAL
	}

	fd.incref()
	defer fd.decref()
	if fd.rdeadline_delta > 0 {
		fd.rdeadline = pollserver.Now() + fd.rdeadline_delta
	} else {
		fd.rdeadline = 0
	}

	// See ../syscall/exec.go for description of ForkLock.
	// It is okay to hold the lock across syscall.Accept
	// because we have put fd.sysfd into non-blocking mode.
	syscall.ForkLock.RLock()
	var s, e int
	var rsa syscall.Sockaddr
	for {
		if fd.closing {
			syscall.ForkLock.RUnlock()
			return nil, os.EINVAL
		}
		s, rsa, e = syscall.Accept(fd.sysfd)
		if e != syscall.EAGAIN || fd.rdeadline < 0 {
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

	if nfd, err = newFD(s, fd.family, fd.proto, fd.net); err != nil {
		syscall.Close(s)
		return nil, err
	}
	lsa, _ := syscall.Getsockname(nfd.sysfd)
	nfd.setAddr(toAddr(lsa), toAddr(rsa))
	return nfd, nil
}

func (fd *netFD) dup() (f *os.File, err os.Error) {
	ns, e := syscall.Dup(fd.sysfd)
	if e != 0 {
		return nil, &OpError{"dup", fd.net, fd.laddr, os.Errno(e)}
	}

	// We want blocking mode for the new fd, hence the double negative.
	if e = syscall.SetNonblock(ns, false); e != 0 {
		return nil, &OpError{"setnonblock", fd.net, fd.laddr, os.Errno(e)}
	}

	return os.NewFile(ns, fd.sysfile.Name()), nil
}

func closesocket(s int) (errno int) {
	return syscall.Close(s)
}

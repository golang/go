// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd

package net

import (
	"io"
	"os"
	"runtime"
	"sync"
	"syscall"
	"time"
)

// Network file descriptor.
type netFD struct {
	// locking/lifetime of sysfd
	sysmu  sync.Mutex
	sysref int

	// must lock both sysmu and pollserver to write
	// can lock either to read
	closing bool

	// immutable until Close
	sysfd       int
	family      int
	sotype      int
	isConnected bool
	sysfile     *os.File
	cr          chan error
	cw          chan error
	net         string
	laddr       Addr
	raddr       Addr

	// serialize access to Read and Write methods
	rio, wio sync.Mutex

	// read and write deadlines
	rdeadline, wdeadline deadline

	// owned by fd wait server
	ncr, ncw int

	// wait server
	pollServer *pollServer
}

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
func (s *pollServer) Evict(fd *netFD) {
	if s.pending[fd.sysfd<<1] == fd {
		s.WakeFD(fd, 'r', errClosing)
		s.poll.DelFD(fd.sysfd, 'r')
		delete(s.pending, fd.sysfd<<1)
	}
	if s.pending[fd.sysfd<<1|1] == fd {
		s.WakeFD(fd, 'w', errClosing)
		s.poll.DelFD(fd.sysfd, 'w')
		delete(s.pending, fd.sysfd<<1|1)
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
				if mode == 'r' {
					s.poll.DelFD(fd.sysfd, mode)
				} else {
					s.poll.DelFD(fd.sysfd, mode)
				}
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

// Network FD methods.
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

func server(fd int) *pollServer {
	pollN := runtime.GOMAXPROCS(0)
	if pollN > pollMaxN {
		pollN = pollMaxN
	}
	k := fd % pollN
	startServersOnce[k]()
	return pollservers[k]
}

func dialTimeout(net, addr string, timeout time.Duration) (Conn, error) {
	deadline := time.Now().Add(timeout)
	_, addri, err := resolveNetAddr("dial", net, addr, deadline)
	if err != nil {
		return nil, err
	}
	return dialAddr(net, addr, addri, deadline)
}

func newFD(fd, family, sotype int, net string) (*netFD, error) {
	netfd := &netFD{
		sysfd:  fd,
		family: family,
		sotype: sotype,
		net:    net,
	}
	netfd.cr = make(chan error, 1)
	netfd.cw = make(chan error, 1)
	netfd.pollServer = server(fd)
	return netfd, nil
}

func (fd *netFD) setAddr(laddr, raddr Addr) {
	fd.laddr = laddr
	fd.raddr = raddr
	fd.sysfile = os.NewFile(uintptr(fd.sysfd), fd.net)
}

func (fd *netFD) name() string {
	var ls, rs string
	if fd.laddr != nil {
		ls = fd.laddr.String()
	}
	if fd.raddr != nil {
		rs = fd.raddr.String()
	}
	return fd.net + ":" + ls + "->" + rs
}

func (fd *netFD) connect(ra syscall.Sockaddr) error {
	err := syscall.Connect(fd.sysfd, ra)
	if err == syscall.EINPROGRESS {
		if err = fd.pollServer.WaitWrite(fd); err != nil {
			return err
		}
		var e int
		e, err = syscall.GetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_ERROR)
		if err != nil {
			return os.NewSyscallError("getsockopt", err)
		}
		if e != 0 {
			err = syscall.Errno(e)
		}
	}
	return err
}

// Add a reference to this fd.
// If closing==true, pollserver must be locked; mark the fd as closing.
// Returns an error if the fd cannot be used.
func (fd *netFD) incref(closing bool) error {
	fd.sysmu.Lock()
	if fd.closing {
		fd.sysmu.Unlock()
		return errClosing
	}
	fd.sysref++
	if closing {
		fd.closing = true
	}
	fd.sysmu.Unlock()
	return nil
}

// Remove a reference to this FD and close if we've been asked to do so (and
// there are no references left.
func (fd *netFD) decref() {
	fd.sysmu.Lock()
	fd.sysref--
	if fd.closing && fd.sysref == 0 && fd.sysfile != nil {
		fd.sysfile.Close()
		fd.sysfile = nil
		fd.sysfd = -1
	}
	fd.sysmu.Unlock()
}

func (fd *netFD) Close() error {
	fd.pollServer.Lock() // needed for both fd.incref(true) and pollserver.Evict
	if err := fd.incref(true); err != nil {
		fd.pollServer.Unlock()
		return err
	}
	// Unblock any I/O.  Once it all unblocks and returns,
	// so that it cannot be referring to fd.sysfd anymore,
	// the final decref will close fd.sysfd.  This should happen
	// fairly quickly, since all the I/O is non-blocking, and any
	// attempts to block in the pollserver will return errClosing.
	fd.pollServer.Evict(fd)
	fd.pollServer.Unlock()
	fd.decref()
	return nil
}

func (fd *netFD) shutdown(how int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.Shutdown(fd.sysfd, how)
	if err != nil {
		return &OpError{"shutdown", fd.net, fd.laddr, err}
	}
	return nil
}

func (fd *netFD) CloseRead() error {
	return fd.shutdown(syscall.SHUT_RD)
}

func (fd *netFD) CloseWrite() error {
	return fd.shutdown(syscall.SHUT_WR)
}

func (fd *netFD) Read(p []byte) (n int, err error) {
	fd.rio.Lock()
	defer fd.rio.Unlock()
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	for {
		if fd.rdeadline.expired() {
			err = errTimeout
			break
		}
		n, err = syscall.Read(int(fd.sysfd), p)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN {
				if err = fd.pollServer.WaitRead(fd); err == nil {
					continue
				}
			}
		}
		err = chkReadErr(n, err, fd)
		break
	}
	if err != nil && err != io.EOF {
		err = &OpError{"read", fd.net, fd.raddr, err}
	}
	return
}

func (fd *netFD) ReadFrom(p []byte) (n int, sa syscall.Sockaddr, err error) {
	fd.rio.Lock()
	defer fd.rio.Unlock()
	if err := fd.incref(false); err != nil {
		return 0, nil, err
	}
	defer fd.decref()
	for {
		if fd.rdeadline.expired() {
			err = errTimeout
			break
		}
		n, sa, err = syscall.Recvfrom(fd.sysfd, p, 0)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN {
				if err = fd.pollServer.WaitRead(fd); err == nil {
					continue
				}
			}
		}
		err = chkReadErr(n, err, fd)
		break
	}
	if err != nil && err != io.EOF {
		err = &OpError{"read", fd.net, fd.laddr, err}
	}
	return
}

func (fd *netFD) ReadMsg(p []byte, oob []byte) (n, oobn, flags int, sa syscall.Sockaddr, err error) {
	fd.rio.Lock()
	defer fd.rio.Unlock()
	if err := fd.incref(false); err != nil {
		return 0, 0, 0, nil, err
	}
	defer fd.decref()
	for {
		if fd.rdeadline.expired() {
			err = errTimeout
			break
		}
		n, oobn, flags, sa, err = syscall.Recvmsg(fd.sysfd, p, oob, 0)
		if err != nil {
			// TODO(dfc) should n and oobn be set to 0
			if err == syscall.EAGAIN {
				if err = fd.pollServer.WaitRead(fd); err == nil {
					continue
				}
			}
		}
		err = chkReadErr(n, err, fd)
		break
	}
	if err != nil && err != io.EOF {
		err = &OpError{"read", fd.net, fd.laddr, err}
	}
	return
}

func chkReadErr(n int, err error, fd *netFD) error {
	if n == 0 && err == nil && fd.sotype != syscall.SOCK_DGRAM && fd.sotype != syscall.SOCK_RAW {
		return io.EOF
	}
	return err
}

func (fd *netFD) Write(p []byte) (nn int, err error) {
	fd.wio.Lock()
	defer fd.wio.Unlock()
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	for {
		if fd.wdeadline.expired() {
			err = errTimeout
			break
		}
		var n int
		n, err = syscall.Write(int(fd.sysfd), p[nn:])
		if n > 0 {
			nn += n
		}
		if nn == len(p) {
			break
		}
		if err == syscall.EAGAIN {
			if err = fd.pollServer.WaitWrite(fd); err == nil {
				continue
			}
		}
		if err != nil {
			n = 0
			break
		}
		if n == 0 {
			err = io.ErrUnexpectedEOF
			break
		}
	}
	if err != nil {
		err = &OpError{"write", fd.net, fd.raddr, err}
	}
	return nn, err
}

func (fd *netFD) WriteTo(p []byte, sa syscall.Sockaddr) (n int, err error) {
	fd.wio.Lock()
	defer fd.wio.Unlock()
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	for {
		if fd.wdeadline.expired() {
			err = errTimeout
			break
		}
		err = syscall.Sendto(fd.sysfd, p, 0, sa)
		if err == syscall.EAGAIN {
			if err = fd.pollServer.WaitWrite(fd); err == nil {
				continue
			}
		}
		break
	}
	if err == nil {
		n = len(p)
	} else {
		err = &OpError{"write", fd.net, fd.raddr, err}
	}
	return
}

func (fd *netFD) WriteMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	fd.wio.Lock()
	defer fd.wio.Unlock()
	if err := fd.incref(false); err != nil {
		return 0, 0, err
	}
	defer fd.decref()
	for {
		if fd.wdeadline.expired() {
			err = errTimeout
			break
		}
		err = syscall.Sendmsg(fd.sysfd, p, oob, sa, 0)
		if err == syscall.EAGAIN {
			if err = fd.pollServer.WaitWrite(fd); err == nil {
				continue
			}
		}
		break
	}
	if err == nil {
		n = len(p)
		oobn = len(oob)
	} else {
		err = &OpError{"write", fd.net, fd.raddr, err}
	}
	return
}

func (fd *netFD) accept(toAddr func(syscall.Sockaddr) Addr) (netfd *netFD, err error) {
	if err := fd.incref(false); err != nil {
		return nil, err
	}
	defer fd.decref()

	var s int
	var rsa syscall.Sockaddr
	for {
		s, rsa, err = accept(fd.sysfd)
		if err != nil {
			if err == syscall.EAGAIN {
				if err = fd.pollServer.WaitRead(fd); err == nil {
					continue
				}
			} else if err == syscall.ECONNABORTED {
				// This means that a socket on the listen queue was closed
				// before we Accept()ed it; it's a silly error, so try again.
				continue
			}
			return nil, &OpError{"accept", fd.net, fd.laddr, err}
		}
		break
	}

	if netfd, err = newFD(s, fd.family, fd.sotype, fd.net); err != nil {
		closesocket(s)
		return nil, err
	}
	lsa, _ := syscall.Getsockname(netfd.sysfd)
	netfd.setAddr(toAddr(lsa), toAddr(rsa))
	return netfd, nil
}

func (fd *netFD) dup() (f *os.File, err error) {
	syscall.ForkLock.RLock()
	ns, err := syscall.Dup(fd.sysfd)
	if err != nil {
		syscall.ForkLock.RUnlock()
		return nil, &OpError{"dup", fd.net, fd.laddr, err}
	}
	syscall.CloseOnExec(ns)
	syscall.ForkLock.RUnlock()

	// We want blocking mode for the new fd, hence the double negative.
	if err = syscall.SetNonblock(ns, false); err != nil {
		return nil, &OpError{"setnonblock", fd.net, fd.laddr, err}
	}

	return os.NewFile(uintptr(ns), fd.name()), nil
}

func closesocket(s int) error {
	return syscall.Close(s)
}

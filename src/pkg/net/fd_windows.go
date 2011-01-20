// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"runtime"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// IO completion result parameters.
type ioResult struct {
	key   uint32
	qty   uint32
	errno int
}

// Network file descriptor.
type netFD struct {
	// locking/lifetime of sysfd
	sysmu   sync.Mutex
	sysref  int
	closing bool

	// immutable until Close
	sysfd  int
	family int
	proto  int
	cr     chan *ioResult
	cw     chan *ioResult
	net    string
	laddr  Addr
	raddr  Addr

	// owned by client
	rdeadline_delta int64
	rdeadline       int64
	rio             sync.Mutex
	wdeadline_delta int64
	wdeadline       int64
	wio             sync.Mutex
}

type InvalidConnError struct{}

func (e *InvalidConnError) String() string  { return "invalid net.Conn" }
func (e *InvalidConnError) Temporary() bool { return false }
func (e *InvalidConnError) Timeout() bool   { return false }

// pollServer will run around waiting for io completion request
// to arrive. Every request received will contain channel to signal
// io owner about the completion.

type pollServer struct {
	iocp int32
}

func newPollServer() (s *pollServer, err os.Error) {
	s = new(pollServer)
	var e int
	if s.iocp, e = syscall.CreateIoCompletionPort(-1, 0, 0, 1); e != 0 {
		return nil, os.NewSyscallError("CreateIoCompletionPort", e)
	}
	go s.Run()
	return s, nil
}

type ioPacket struct {
	// Used by IOCP interface,
	// it must be first field of the struct,
	// as our code rely on it.
	o syscall.Overlapped

	// Link to the io owner.
	c chan *ioResult

	w *syscall.WSABuf
}

func (s *pollServer) getCompletedIO() (ov *syscall.Overlapped, result *ioResult, err os.Error) {
	var r ioResult
	var o *syscall.Overlapped
	_, e := syscall.GetQueuedCompletionStatus(s.iocp, &r.qty, &r.key, &o, syscall.INFINITE)
	switch {
	case e == 0:
		// Dequeued successfully completed io packet.
		return o, &r, nil
	case e == syscall.WAIT_TIMEOUT && o == nil:
		// Wait has timed out (should not happen now, but might be used in the future).
		return nil, &r, os.NewSyscallError("GetQueuedCompletionStatus", e)
	case o == nil:
		// Failed to dequeue anything -> report the error.
		return nil, &r, os.NewSyscallError("GetQueuedCompletionStatus", e)
	default:
		// Dequeued failed io packet.
		r.errno = e
		return o, &r, nil
	}
	return
}

func (s *pollServer) Run() {
	for {
		o, r, err := s.getCompletedIO()
		if err != nil {
			panic("Run pollServer: " + err.String() + "\n")
		}
		p := (*ioPacket)(unsafe.Pointer(o))
		p.c <- r
	}
}

// Network FD methods.
// All the network FDs use a single pollServer.

var pollserver *pollServer
var onceStartServer sync.Once

func startServer() {
	p, err := newPollServer()
	if err != nil {
		panic("Start pollServer: " + err.String() + "\n")
	}
	pollserver = p

	go timeoutIO()
}

var initErr os.Error

func newFD(fd, family, proto int, net string, laddr, raddr Addr) (f *netFD, err os.Error) {
	if initErr != nil {
		return nil, initErr
	}
	onceStartServer.Do(startServer)
	// Associate our socket with pollserver.iocp.
	if _, e := syscall.CreateIoCompletionPort(int32(fd), pollserver.iocp, 0, 0); e != 0 {
		return nil, &OpError{"CreateIoCompletionPort", net, laddr, os.Errno(e)}
	}
	f = &netFD{
		sysfd:  fd,
		family: family,
		proto:  proto,
		cr:     make(chan *ioResult, 1),
		cw:     make(chan *ioResult, 1),
		net:    net,
		laddr:  laddr,
		raddr:  raddr,
	}
	runtime.SetFinalizer(f, (*netFD).Close)
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
		closesocket(fd.sysfd)
		fd.sysfd = -1
		// no need for a finalizer anymore
		runtime.SetFinalizer(fd, nil)
	}
	fd.sysmu.Unlock()
}

func (fd *netFD) Close() os.Error {
	if fd == nil || fd.sysfd == -1 {
		return os.EINVAL
	}

	fd.incref()
	syscall.Shutdown(fd.sysfd, syscall.SHUT_RDWR)
	fd.closing = true
	fd.decref()
	return nil
}

func newWSABuf(p []byte) *syscall.WSABuf {
	var p0 *byte
	if len(p) > 0 {
		p0 = (*byte)(unsafe.Pointer(&p[0]))
	}
	return &syscall.WSABuf{uint32(len(p)), p0}
}

func waitPacket(fd *netFD, pckt *ioPacket, mode int) (r *ioResult) {
	var delta int64
	if mode == 'r' {
		delta = fd.rdeadline_delta
	}
	if mode == 'w' {
		delta = fd.wdeadline_delta
	}
	if delta <= 0 {
		return <-pckt.c
	}

	select {
	case r = <-pckt.c:
	case <-time.After(delta):
		a := &arg{f: cancel, fd: fd, pckt: pckt, c: make(chan int)}
		ioChan <- a
		<-a.c
		r = <-pckt.c
		if r.errno == 995 { // IO Canceled
			r.errno = syscall.EWOULDBLOCK
		}
	}
	return r
}

const (
	read = iota
	readfrom
	write
	writeto
	cancel
)

type arg struct {
	f     int
	fd    *netFD
	pckt  *ioPacket
	done  *uint32
	flags *uint32
	rsa   *syscall.RawSockaddrAny
	size  *int32
	sa    *syscall.Sockaddr
	c     chan int
}

var ioChan chan *arg = make(chan *arg)

func timeoutIO() {
	// CancelIO only cancels all pending input and output (I/O) operations that are
	// issued by the calling thread for the specified file, does not cancel I/O
	// operations that other threads issue for a file handle. So we need do all timeout
	// I/O in single OS thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for {
		o := <-ioChan
		var e int
		switch o.f {
		case read:
			e = syscall.WSARecv(uint32(o.fd.sysfd), o.pckt.w, 1, o.done, o.flags, &o.pckt.o, nil)
		case readfrom:
			e = syscall.WSARecvFrom(uint32(o.fd.sysfd), o.pckt.w, 1, o.done, o.flags, o.rsa, o.size, &o.pckt.o, nil)
		case write:
			e = syscall.WSASend(uint32(o.fd.sysfd), o.pckt.w, 1, o.done, uint32(0), &o.pckt.o, nil)
		case writeto:
			e = syscall.WSASendto(uint32(o.fd.sysfd), o.pckt.w, 1, o.done, 0, *o.sa, &o.pckt.o, nil)
		case cancel:
			_, e = syscall.CancelIo(uint32(o.fd.sysfd))
		}
		o.c <- e
	}
}

func (fd *netFD) Read(p []byte) (n int, err os.Error) {
	if fd == nil {
		return 0, os.EINVAL
	}
	fd.rio.Lock()
	defer fd.rio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.sysfd == -1 {
		return 0, os.EINVAL
	}
	// Submit receive request.
	var pckt ioPacket
	pckt.c = fd.cr
	pckt.w = newWSABuf(p)
	var done uint32
	flags := uint32(0)
	var e int
	if fd.rdeadline_delta > 0 {
		a := &arg{f: read, fd: fd, pckt: &pckt, done: &done, flags: &flags, c: make(chan int)}
		ioChan <- a
		e = <-a.c
	} else {
		e = syscall.WSARecv(uint32(fd.sysfd), pckt.w, 1, &done, &flags, &pckt.o, nil)
	}
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		return 0, &OpError{"WSARecv", fd.net, fd.laddr, os.Errno(e)}
	}
	// Wait for our request to complete.
	r := waitPacket(fd, &pckt, 'r')
	if r.errno != 0 {
		err = &OpError{"WSARecv", fd.net, fd.laddr, os.Errno(r.errno)}
	}
	n = int(r.qty)
	if err == nil && n == 0 {
		err = os.EOF
	}
	return
}

func (fd *netFD) ReadFrom(p []byte) (n int, sa syscall.Sockaddr, err os.Error) {
	if fd == nil {
		return 0, nil, os.EINVAL
	}
	if len(p) == 0 {
		return 0, nil, nil
	}
	fd.rio.Lock()
	defer fd.rio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.sysfd == -1 {
		return 0, nil, os.EINVAL
	}
	// Submit receive request.
	var pckt ioPacket
	pckt.c = fd.cr
	pckt.w = newWSABuf(p)
	var done uint32
	flags := uint32(0)
	var rsa syscall.RawSockaddrAny
	l := int32(unsafe.Sizeof(rsa))
	var e int
	if fd.rdeadline_delta > 0 {
		a := &arg{f: readfrom, fd: fd, pckt: &pckt, done: &done, flags: &flags, rsa: &rsa, size: &l, c: make(chan int)}
		ioChan <- a
		e = <-a.c
	} else {
		e = syscall.WSARecvFrom(uint32(fd.sysfd), pckt.w, 1, &done, &flags, &rsa, &l, &pckt.o, nil)
	}
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		return 0, nil, &OpError{"WSARecvFrom", fd.net, fd.laddr, os.Errno(e)}
	}
	// Wait for our request to complete.
	r := waitPacket(fd, &pckt, 'r')
	if r.errno != 0 {
		err = &OpError{"WSARecvFrom", fd.net, fd.laddr, os.Errno(r.errno)}
	}
	n = int(r.qty)
	sa, _ = rsa.Sockaddr()
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
	if fd.sysfd == -1 {
		return 0, os.EINVAL
	}
	// Submit send request.
	var pckt ioPacket
	pckt.c = fd.cw
	pckt.w = newWSABuf(p)
	var done uint32
	var e int
	if fd.wdeadline_delta > 0 {
		a := &arg{f: write, fd: fd, pckt: &pckt, done: &done, c: make(chan int)}
		ioChan <- a
		e = <-a.c
	} else {
		e = syscall.WSASend(uint32(fd.sysfd), pckt.w, 1, &done, uint32(0), &pckt.o, nil)
	}
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		return 0, &OpError{"WSASend", fd.net, fd.laddr, os.Errno(e)}
	}
	// Wait for our request to complete.
	r := waitPacket(fd, &pckt, 'w')
	if r.errno != 0 {
		err = &OpError{"WSASend", fd.net, fd.laddr, os.Errno(r.errno)}
	}
	n = int(r.qty)
	return
}

func (fd *netFD) WriteTo(p []byte, sa syscall.Sockaddr) (n int, err os.Error) {
	if fd == nil {
		return 0, os.EINVAL
	}
	if len(p) == 0 {
		return 0, nil
	}
	fd.wio.Lock()
	defer fd.wio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.sysfd == -1 {
		return 0, os.EINVAL
	}
	// Submit send request.
	var pckt ioPacket
	pckt.c = fd.cw
	pckt.w = newWSABuf(p)
	var done uint32
	var e int
	if fd.wdeadline_delta > 0 {
		a := &arg{f: writeto, fd: fd, pckt: &pckt, done: &done, sa: &sa, c: make(chan int)}
		ioChan <- a
		e = <-a.c
	} else {
		e = syscall.WSASendto(uint32(fd.sysfd), pckt.w, 1, &done, 0, sa, &pckt.o, nil)
	}
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		return 0, &OpError{"WSASendTo", fd.net, fd.laddr, os.Errno(e)}
	}
	// Wait for our request to complete.
	r := waitPacket(fd, &pckt, 'w')
	if r.errno != 0 {
		err = &OpError{"WSASendTo", fd.net, fd.laddr, os.Errno(r.errno)}
	}
	n = int(r.qty)
	return
}

func (fd *netFD) accept(toAddr func(syscall.Sockaddr) Addr) (nfd *netFD, err os.Error) {
	if fd == nil || fd.sysfd == -1 {
		return nil, os.EINVAL
	}
	fd.incref()
	defer fd.decref()

	// Get new socket.
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, e := syscall.Socket(fd.family, fd.proto, 0)
	if e != 0 {
		syscall.ForkLock.RUnlock()
		return nil, os.Errno(e)
	}
	syscall.CloseOnExec(s)
	syscall.ForkLock.RUnlock()

	// Associate our new socket with IOCP.
	onceStartServer.Do(startServer)
	if _, e = syscall.CreateIoCompletionPort(int32(s), pollserver.iocp, 0, 0); e != 0 {
		return nil, &OpError{"CreateIoCompletionPort", fd.net, fd.laddr, os.Errno(e)}
	}

	// Submit accept request.
	// Will use new unique channel here, because, unlike Read or Write,
	// Accept is expected to be executed by many goroutines simultaniously.
	var pckt ioPacket
	pckt.c = make(chan *ioResult)
	attrs, e := syscall.AcceptIOCP(fd.sysfd, s, &pckt.o)
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		closesocket(s)
		return nil, &OpError{"AcceptEx", fd.net, fd.laddr, os.Errno(e)}
	}

	// Wait for peer connection.
	r := <-pckt.c
	if r.errno != 0 {
		closesocket(s)
		return nil, &OpError{"AcceptEx", fd.net, fd.laddr, os.Errno(r.errno)}
	}

	// Inherit properties of the listening socket.
	e = syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_UPDATE_ACCEPT_CONTEXT, fd.sysfd)
	if e != 0 {
		closesocket(s)
		return nil, &OpError{"Setsockopt", fd.net, fd.laddr, os.Errno(r.errno)}
	}

	// Get local and peer addr out of AcceptEx buffer.
	lsa, rsa := syscall.GetAcceptIOCPSockaddrs(attrs)

	// Create our netFD and return it for further use.
	laddr := toAddr(lsa)
	raddr := toAddr(rsa)

	f := &netFD{
		sysfd:  s,
		family: fd.family,
		proto:  fd.proto,
		cr:     make(chan *ioResult, 1),
		cw:     make(chan *ioResult, 1),
		net:    fd.net,
		laddr:  laddr,
		raddr:  raddr,
	}
	runtime.SetFinalizer(f, (*netFD).Close)
	return f, nil
}

func closesocket(s int) (errno int) {
	return syscall.Closesocket(int32(s))
}

func init() {
	var d syscall.WSAData
	e := syscall.WSAStartup(uint32(0x101), &d)
	if e != 0 {
		initErr = os.NewSyscallError("WSAStartup", e)
	}
}

func (fd *netFD) dup() (f *os.File, err os.Error) {
	// TODO: Implement this
	return nil, os.NewSyscallError("dup", syscall.EWINDOWS)
}

func (fd *netFD) ReadMsg(p []byte, oob []byte) (n, oobn, flags int, sa syscall.Sockaddr, err os.Error) {
	return 0, 0, 0, nil, os.EAFNOSUPPORT
}

func (fd *netFD) WriteMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err os.Error) {
	return 0, 0, os.EAFNOSUPPORT
}

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"sync"
	"syscall"
	"unsafe"
)

// BUG(brainman): The Windows implementation does not implement SetTimeout.

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
	sysfd   int
	family  int
	proto   int
	sysfile *os.File
	cr      chan *ioResult
	cw      chan *ioResult
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
		cr:     make(chan *ioResult),
		cw:     make(chan *ioResult),
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

func newWSABuf(p []byte) *syscall.WSABuf {
	var p0 *byte
	if len(p) > 0 {
		p0 = (*byte)(unsafe.Pointer(&p[0]))
	}
	return &syscall.WSABuf{uint32(len(p)), p0}
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
	// Submit receive request.
	var pckt ioPacket
	pckt.c = fd.cr
	var done uint32
	flags := uint32(0)
	e := syscall.WSARecv(uint32(fd.sysfd), newWSABuf(p), 1, &done, &flags, &pckt.o, nil)
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		return 0, &OpError{"WSARecv", fd.net, fd.laddr, os.Errno(e)}
	}
	// Wait for our request to complete.
	r := <-pckt.c
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
	var r syscall.Sockaddr
	return 0, r, nil
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
	// Submit send request.
	var pckt ioPacket
	pckt.c = fd.cw
	var done uint32
	e := syscall.WSASend(uint32(fd.sysfd), newWSABuf(p), 1, &done, uint32(0), &pckt.o, nil)
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		return 0, &OpError{"WSASend", fd.net, fd.laddr, os.Errno(e)}
	}
	// Wait for our request to complete.
	r := <-pckt.c
	if r.errno != 0 {
		err = &OpError{"WSASend", fd.net, fd.laddr, os.Errno(r.errno)}
	}
	n = int(r.qty)
	return
}

func (fd *netFD) WriteTo(p []byte, sa syscall.Sockaddr) (n int, err os.Error) {
	return 0, nil
}

func (fd *netFD) accept(toAddr func(syscall.Sockaddr) Addr) (nfd *netFD, err os.Error) {
	if fd == nil || fd.sysfile == nil {
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
		syscall.Close(s)
		return nil, &OpError{"AcceptEx", fd.net, fd.laddr, os.Errno(e)}
	}

	// Wait for peer connection.
	r := <-pckt.c
	if r.errno != 0 {
		syscall.Close(s)
		return nil, &OpError{"AcceptEx", fd.net, fd.laddr, os.Errno(r.errno)}
	}

	// Inherit properties of the listening socket.
	e = syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_UPDATE_ACCEPT_CONTEXT, fd.sysfd)
	if e != 0 {
		syscall.Close(s)
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
		cr:     make(chan *ioResult),
		cw:     make(chan *ioResult),
		net:    fd.net,
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
	f.sysfile = os.NewFile(s, fd.net+":"+ls+"->"+rs)
	return f, nil
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

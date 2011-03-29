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

type InvalidConnError struct{}

func (e *InvalidConnError) String() string  { return "invalid net.Conn" }
func (e *InvalidConnError) Temporary() bool { return false }
func (e *InvalidConnError) Timeout() bool   { return false }

var initErr os.Error

func init() {
	var d syscall.WSAData
	e := syscall.WSAStartup(uint32(0x101), &d)
	if e != 0 {
		initErr = os.NewSyscallError("WSAStartup", e)
	}
}

func closesocket(s int) (errno int) {
	return syscall.Closesocket(int32(s))
}

// Interface for all io operations.
type anOpIface interface {
	Op() *anOp
	Name() string
	Submit() (errno int)
}

// IO completion result parameters.
type ioResult struct {
	qty uint32
	err int
}

// anOp implements functionality common to all io operations.
type anOp struct {
	// Used by IOCP interface, it must be first field
	// of the struct, as our code rely on it.
	o syscall.Overlapped

	resultc chan ioResult // io completion results
	errnoc  chan int      // io submit / cancel operation errors
	fd      *netFD
}

func (o *anOp) Init(fd *netFD) {
	o.fd = fd
	o.resultc = make(chan ioResult, 1)
	o.errnoc = make(chan int)
}

func (o *anOp) Op() *anOp {
	return o
}

// bufOp is used by io operations that read / write
// data from / to client buffer.
type bufOp struct {
	anOp
	buf syscall.WSABuf
}

func (o *bufOp) Init(fd *netFD, buf []byte) {
	o.anOp.Init(fd)
	o.buf.Len = uint32(len(buf))
	if len(buf) == 0 {
		o.buf.Buf = nil
	} else {
		o.buf.Buf = (*byte)(unsafe.Pointer(&buf[0]))
	}
}

// resultSrv will retreive all io completion results from
// iocp and send them to the correspondent waiting client
// goroutine via channel supplied in the request.
type resultSrv struct {
	iocp int32
}

func (s *resultSrv) Run() {
	var o *syscall.Overlapped
	var key uint32
	var r ioResult
	for {
		r.err = syscall.GetQueuedCompletionStatus(s.iocp, &(r.qty), &key, &o, syscall.INFINITE)
		switch {
		case r.err == 0:
			// Dequeued successfully completed io packet.
		case r.err == syscall.WAIT_TIMEOUT && o == nil:
			// Wait has timed out (should not happen now, but might be used in the future).
			panic("GetQueuedCompletionStatus timed out")
		case o == nil:
			// Failed to dequeue anything -> report the error.
			panic("GetQueuedCompletionStatus failed " + syscall.Errstr(r.err))
		default:
			// Dequeued failed io packet.
		}
		(*anOp)(unsafe.Pointer(o)).resultc <- r
	}
}


// ioSrv executes net io requests.
type ioSrv struct {
	submchan chan anOpIface // submit io requests
	canchan  chan anOpIface // cancel io requests
}

// ProcessRemoteIO will execute submit io requests on behalf
// of other goroutines, all on a single os thread, so it can
// cancel them later. Results of all operations will be sent
// back to their requesters via channel supplied in request.
func (s *ioSrv) ProcessRemoteIO() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for {
		select {
		case o := <-s.submchan:
			o.Op().errnoc <- o.Submit()
		case o := <-s.canchan:
			o.Op().errnoc <- syscall.CancelIo(uint32(o.Op().fd.sysfd))
		}
	}
}

// ExecIO executes a single io operation. It either executes it
// inline, or, if timeouts are employed, passes the request onto
// a special goroutine and waits for completion or cancels request.
func (s *ioSrv) ExecIO(oi anOpIface, deadline_delta int64) (n int, err os.Error) {
	var e int
	o := oi.Op()
	if deadline_delta > 0 {
		// Send request to a special dedicated thread,
		// so it can stop the io with CancelIO later.
		s.submchan <- oi
		e = <-o.errnoc
	} else {
		e = oi.Submit()
	}
	switch e {
	case 0:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for it's completion.
	default:
		return 0, &OpError{oi.Name(), o.fd.net, o.fd.laddr, os.Errno(e)}
	}
	// Wait for our request to complete.
	var r ioResult
	if deadline_delta > 0 {
		select {
		case r = <-o.resultc:
		case <-time.After(deadline_delta):
			s.canchan <- oi
			<-o.errnoc
			r = <-o.resultc
			if r.err == syscall.ERROR_OPERATION_ABORTED { // IO Canceled
				r.err = syscall.EWOULDBLOCK
			}
		}
	} else {
		r = <-o.resultc
	}
	if r.err != 0 {
		err = &OpError{oi.Name(), o.fd.net, o.fd.laddr, os.Errno(r.err)}
	}
	return int(r.qty), err
}

// Start helper goroutines.
var resultsrv *resultSrv
var iosrv *ioSrv
var onceStartServer sync.Once

func startServer() {
	resultsrv = new(resultSrv)
	var errno int
	resultsrv.iocp, errno = syscall.CreateIoCompletionPort(-1, 0, 0, 1)
	if errno != 0 {
		panic("CreateIoCompletionPort failed " + syscall.Errstr(errno))
	}
	go resultsrv.Run()

	iosrv = new(ioSrv)
	iosrv.submchan = make(chan anOpIface)
	iosrv.canchan = make(chan anOpIface)
	go iosrv.ProcessRemoteIO()
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

func allocFD(fd, family, proto int, net string) (f *netFD) {
	f = &netFD{
		sysfd:  fd,
		family: family,
		proto:  proto,
		net:    net,
	}
	runtime.SetFinalizer(f, (*netFD).Close)
	return f
}

func newFD(fd, family, proto int, net string) (f *netFD, err os.Error) {
	if initErr != nil {
		return nil, initErr
	}
	onceStartServer.Do(startServer)
	// Associate our socket with resultsrv.iocp.
	if _, e := syscall.CreateIoCompletionPort(int32(fd), resultsrv.iocp, 0, 0); e != 0 {
		return nil, os.Errno(e)
	}
	return allocFD(fd, family, proto, net), nil
}

func (fd *netFD) setAddr(laddr, raddr Addr) {
	fd.laddr = laddr
	fd.raddr = raddr
}

func (fd *netFD) connect(ra syscall.Sockaddr) (err os.Error) {
	e := syscall.Connect(fd.sysfd, ra)
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
		// use the resultsrv for Close too.  Sigh.
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

// Read from network.

type readOp struct {
	bufOp
}

func (o *readOp) Submit() (errno int) {
	var d, f uint32
	return syscall.WSARecv(uint32(o.fd.sysfd), &o.buf, 1, &d, &f, &o.o, nil)
}

func (o *readOp) Name() string {
	return "WSARecv"
}

func (fd *netFD) Read(buf []byte) (n int, err os.Error) {
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
	var o readOp
	o.Init(fd, buf)
	n, err = iosrv.ExecIO(&o, fd.rdeadline_delta)
	if err == nil && n == 0 {
		err = os.EOF
	}
	return
}

// ReadFrom from network.

type readFromOp struct {
	bufOp
	rsa syscall.RawSockaddrAny
}

func (o *readFromOp) Submit() (errno int) {
	var d, f uint32
	l := int32(unsafe.Sizeof(o.rsa))
	return syscall.WSARecvFrom(uint32(o.fd.sysfd), &o.buf, 1, &d, &f, &o.rsa, &l, &o.o, nil)
}

func (o *readFromOp) Name() string {
	return "WSARecvFrom"
}

func (fd *netFD) ReadFrom(buf []byte) (n int, sa syscall.Sockaddr, err os.Error) {
	if fd == nil {
		return 0, nil, os.EINVAL
	}
	if len(buf) == 0 {
		return 0, nil, nil
	}
	fd.rio.Lock()
	defer fd.rio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.sysfd == -1 {
		return 0, nil, os.EINVAL
	}
	var o readFromOp
	o.Init(fd, buf)
	n, err = iosrv.ExecIO(&o, fd.rdeadline_delta)
	sa, _ = o.rsa.Sockaddr()
	return
}

// Write to network.

type writeOp struct {
	bufOp
}

func (o *writeOp) Submit() (errno int) {
	var d uint32
	return syscall.WSASend(uint32(o.fd.sysfd), &o.buf, 1, &d, 0, &o.o, nil)
}

func (o *writeOp) Name() string {
	return "WSASend"
}

func (fd *netFD) Write(buf []byte) (n int, err os.Error) {
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
	var o writeOp
	o.Init(fd, buf)
	return iosrv.ExecIO(&o, fd.wdeadline_delta)
}

// WriteTo to network.

type writeToOp struct {
	bufOp
	sa syscall.Sockaddr
}

func (o *writeToOp) Submit() (errno int) {
	var d uint32
	return syscall.WSASendto(uint32(o.fd.sysfd), &o.buf, 1, &d, 0, o.sa, &o.o, nil)
}

func (o *writeToOp) Name() string {
	return "WSASendto"
}

func (fd *netFD) WriteTo(buf []byte, sa syscall.Sockaddr) (n int, err os.Error) {
	if fd == nil {
		return 0, os.EINVAL
	}
	if len(buf) == 0 {
		return 0, nil
	}
	fd.wio.Lock()
	defer fd.wio.Unlock()
	fd.incref()
	defer fd.decref()
	if fd.sysfd == -1 {
		return 0, os.EINVAL
	}
	var o writeToOp
	o.Init(fd, buf)
	o.sa = sa
	return iosrv.ExecIO(&o, fd.wdeadline_delta)
}

// Accept new network connections.

type acceptOp struct {
	anOp
	newsock int
	attrs   [2]syscall.RawSockaddrAny // space for local and remote address only
}

func (o *acceptOp) Submit() (errno int) {
	var d uint32
	l := uint32(unsafe.Sizeof(o.attrs[0]))
	return syscall.AcceptEx(uint32(o.fd.sysfd), uint32(o.newsock),
		(*byte)(unsafe.Pointer(&o.attrs[0])), 0, l, l, &d, &o.o)
}

func (o *acceptOp) Name() string {
	return "AcceptEx"
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
	if _, e = syscall.CreateIoCompletionPort(int32(s), resultsrv.iocp, 0, 0); e != 0 {
		return nil, &OpError{"CreateIoCompletionPort", fd.net, fd.laddr, os.Errno(e)}
	}

	// Submit accept request.
	var o acceptOp
	o.Init(fd)
	o.newsock = s
	_, err = iosrv.ExecIO(&o, 0)
	if err != nil {
		closesocket(s)
		return nil, err
	}

	// Inherit properties of the listening socket.
	e = syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_UPDATE_ACCEPT_CONTEXT, fd.sysfd)
	if e != 0 {
		closesocket(s)
		return nil, err
	}

	// Get local and peer addr out of AcceptEx buffer.
	var lrsa, rrsa *syscall.RawSockaddrAny
	var llen, rlen int32
	l := uint32(unsafe.Sizeof(*lrsa))
	syscall.GetAcceptExSockaddrs((*byte)(unsafe.Pointer(&o.attrs[0])),
		0, l, l, &lrsa, &llen, &rrsa, &rlen)
	lsa, _ := lrsa.Sockaddr()
	rsa, _ := rrsa.Sockaddr()

	nfd = allocFD(s, fd.family, fd.proto, fd.net)
	nfd.setAddr(toAddr(lsa), toAddr(rsa))
	return nfd, nil
}

// Not implemeted functions.

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

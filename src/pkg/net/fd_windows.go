// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"io"
	"os"
	"runtime"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

var initErr error

// CancelIo Windows API cancels all outstanding IO for a particular
// socket on current thread. To overcome that limitation, we run
// special goroutine, locked to OS single thread, that both starts
// and cancels IO. It means, there are 2 unavoidable thread switches
// for every IO.
// Some newer versions of Windows has new CancelIoEx API, that does
// not have that limitation and can be used from any thread. This
// package uses CancelIoEx API, if present, otherwise it fallback
// to CancelIo.

var canCancelIO bool // determines if CancelIoEx API is present

func sysInit() {
	var d syscall.WSAData
	e := syscall.WSAStartup(uint32(0x202), &d)
	if e != nil {
		initErr = os.NewSyscallError("WSAStartup", e)
	}
	canCancelIO = syscall.LoadCancelIoEx() == nil
	if syscall.LoadGetAddrInfo() == nil {
		lookupPort = newLookupPort
		lookupIP = newLookupIP
	}
}

func closesocket(s syscall.Handle) error {
	return syscall.Closesocket(s)
}

func canUseConnectEx(net string) bool {
	if net == "udp" || net == "udp4" || net == "udp6" {
		// ConnectEx windows API does not support connectionless sockets.
		return false
	}
	return syscall.LoadConnectEx() == nil
}

func resolveAndDial(net, addr string, localAddr Addr, deadline time.Time) (Conn, error) {
	if !canUseConnectEx(net) {
		// Use the relatively inefficient goroutine-racing
		// implementation of DialTimeout.
		return resolveAndDialChannel(net, addr, localAddr, deadline)
	}
	ra, err := resolveAddr("dial", net, addr, deadline)
	if err != nil {
		return nil, err
	}
	return dial(net, addr, localAddr, ra, deadline)
}

// Interface for all IO operations.
type anOpIface interface {
	Op() *anOp
	Name() string
	Submit() error
}

// anOp implements functionality common to all IO operations.
// Its beginning must be the same as runtime.net_anOp. Keep these in sync.
type anOp struct {
	// Used by IOCP interface, it must be first field
	// of the struct, as our code rely on it.
	o syscall.Overlapped

	// fields used by runtime.netpoll
	runtimeCtx uintptr
	mode       int32
	errno      int32
	qty        uint32

	errnoc chan error
	fd     *netFD
}

func (o *anOp) Init(fd *netFD, mode int32) {
	o.fd = fd
	o.mode = mode
	o.runtimeCtx = fd.pd.runtimeCtx
	if !canCancelIO {
		var i int
		if mode == 'r' {
			i = 0
		} else {
			i = 1
		}
		if fd.errnoc[i] == nil {
			fd.errnoc[i] = make(chan error)
		}
		o.errnoc = fd.errnoc[i]
	}
}

func (o *anOp) Op() *anOp {
	return o
}

// bufOp is used by IO operations that read / write
// data from / to client buffer.
type bufOp struct {
	anOp
	buf syscall.WSABuf
}

func (o *bufOp) Init(fd *netFD, buf []byte, mode int32) {
	o.anOp.Init(fd, mode)
	o.buf.Len = uint32(len(buf))
	if len(buf) == 0 {
		o.buf.Buf = nil
	} else {
		o.buf.Buf = (*byte)(unsafe.Pointer(&buf[0]))
	}
}

// ioSrv executes net IO requests.
type ioSrv struct {
	submchan chan anOpIface // submit IO requests
	canchan  chan anOpIface // cancel IO requests
}

// ProcessRemoteIO will execute submit IO requests on behalf
// of other goroutines, all on a single os thread, so it can
// cancel them later. Results of all operations will be sent
// back to their requesters via channel supplied in request.
// It is used only when the CancelIoEx API is unavailable.
func (s *ioSrv) ProcessRemoteIO() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for {
		select {
		case o := <-s.submchan:
			o.Op().errnoc <- o.Submit()
		case o := <-s.canchan:
			o.Op().errnoc <- syscall.CancelIo(syscall.Handle(o.Op().fd.sysfd))
		}
	}
}

// ExecIO executes a single IO operation oi. It submits and cancels
// IO in the current thread for systems where Windows CancelIoEx API
// is available. Alternatively, it passes the request onto
// runtime netpoll and waits for completion or cancels request.
func (s *ioSrv) ExecIO(oi anOpIface) (int, error) {
	var err error
	o := oi.Op()
	// Notify runtime netpoll about starting IO.
	err = o.fd.pd.Prepare(int(o.mode))
	if err != nil {
		return 0, &OpError{oi.Name(), o.fd.net, o.fd.laddr, err}
	}
	// Start IO.
	if canCancelIO {
		err = oi.Submit()
	} else {
		// Send request to a special dedicated thread,
		// so it can stop the IO with CancelIO later.
		s.submchan <- oi
		err = <-o.errnoc
	}
	switch err {
	case nil:
		// IO completed immediately, but we need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for its completion.
		err = nil
	default:
		return 0, &OpError{oi.Name(), o.fd.net, o.fd.laddr, err}
	}
	// Wait for our request to complete.
	err = o.fd.pd.Wait(int(o.mode))
	if err == nil {
		// All is good. Extract our IO results and return.
		if o.errno != 0 {
			err = syscall.Errno(o.errno)
			return 0, &OpError{oi.Name(), o.fd.net, o.fd.laddr, err}
		}
		return int(o.qty), nil
	}
	// IO is interrupted by "close" or "timeout"
	netpollErr := err
	switch netpollErr {
	case errClosing, errTimeout:
		// will deal with those.
	default:
		panic("net: unexpected runtime.netpoll error: " + netpollErr.Error())
	}
	// Cancel our request.
	if canCancelIO {
		err := syscall.CancelIoEx(syscall.Handle(o.Op().fd.sysfd), &o.o)
		// Assuming ERROR_NOT_FOUND is returned, if IO is completed.
		if err != nil && err != syscall.ERROR_NOT_FOUND {
			// TODO(brainman): maybe do something else, but panic.
			panic(err)
		}
	} else {
		s.canchan <- oi
		<-o.errnoc
	}
	// Wait for cancellation to complete.
	o.fd.pd.WaitCanceled(int(o.mode))
	if o.errno != 0 {
		err = syscall.Errno(o.errno)
		if err == syscall.ERROR_OPERATION_ABORTED { // IO Canceled
			err = netpollErr
		}
		return 0, &OpError{oi.Name(), o.fd.net, o.fd.laddr, err}
	}
	// We issued cancellation request. But, it seems, IO operation succeeded
	// before cancellation request run. We need to treat IO operation as
	// succeeded (the bytes are actually sent/recv from network).
	return int(o.qty), nil
}

// Start helper goroutines.
var iosrv *ioSrv
var onceStartServer sync.Once

func startServer() {
	iosrv = new(ioSrv)
	if !canCancelIO {
		// Only CancelIo API is available. Lets start special goroutine
		// locked to an OS thread, that both starts and cancels IO.
		iosrv.submchan = make(chan anOpIface)
		iosrv.canchan = make(chan anOpIface)
		go iosrv.ProcessRemoteIO()
	}
}

// Network file descriptor.
type netFD struct {
	// locking/lifetime of sysfd
	sysmu   sync.Mutex
	sysref  int
	closing bool

	// immutable until Close
	sysfd       syscall.Handle
	family      int
	sotype      int
	isConnected bool
	net         string
	laddr       Addr
	raddr       Addr
	errnoc      [2]chan error // read/write submit or cancel operation errors

	// serialize access to Read and Write methods
	rio, wio sync.Mutex

	// wait server
	pd pollDesc
}

func newFD(fd syscall.Handle, family, sotype int, net string) (*netFD, error) {
	if initErr != nil {
		return nil, initErr
	}
	onceStartServer.Do(startServer)
	netfd := &netFD{
		sysfd:  fd,
		family: family,
		sotype: sotype,
		net:    net,
	}
	if err := netfd.pd.Init(netfd); err != nil {
		return nil, err
	}
	return netfd, nil
}

func (fd *netFD) setAddr(laddr, raddr Addr) {
	fd.laddr = laddr
	fd.raddr = raddr
	runtime.SetFinalizer(fd, (*netFD).Close)
}

// Make new connection.

type connectOp struct {
	anOp
	ra syscall.Sockaddr
}

func (o *connectOp) Submit() error {
	return syscall.ConnectEx(o.fd.sysfd, o.ra, nil, 0, nil, &o.o)
}

func (o *connectOp) Name() string {
	return "ConnectEx"
}

func (fd *netFD) connect(la, ra syscall.Sockaddr) error {
	if !canUseConnectEx(fd.net) {
		return syscall.Connect(fd.sysfd, ra)
	}
	// ConnectEx windows API requires an unconnected, previously bound socket.
	if la == nil {
		switch ra.(type) {
		case *syscall.SockaddrInet4:
			la = &syscall.SockaddrInet4{}
		case *syscall.SockaddrInet6:
			la = &syscall.SockaddrInet6{}
		default:
			panic("unexpected type in connect")
		}
		if err := syscall.Bind(fd.sysfd, la); err != nil {
			return err
		}
	}
	// Call ConnectEx API.
	var o connectOp
	o.Init(fd, 'w')
	o.ra = ra
	_, err := iosrv.ExecIO(&o)
	if err != nil {
		return err
	}
	// Refresh socket properties.
	return syscall.Setsockopt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_UPDATE_CONNECT_CONTEXT, (*byte)(unsafe.Pointer(&fd.sysfd)), int32(unsafe.Sizeof(fd.sysfd)))
}

// Add a reference to this fd.
// If closing==true, mark the fd as closing.
// Returns an error if the fd cannot be used.
func (fd *netFD) incref(closing bool) error {
	if fd == nil {
		return errClosing
	}
	fd.sysmu.Lock()
	if fd.closing {
		fd.sysmu.Unlock()
		return errClosing
	}
	fd.sysref++
	if closing {
		fd.closing = true
	}
	closing = fd.closing
	fd.sysmu.Unlock()
	return nil
}

// Remove a reference to this FD and close if we've been asked to do so (and
// there are no references left.
func (fd *netFD) decref() {
	if fd == nil {
		return
	}
	fd.sysmu.Lock()
	fd.sysref--
	if fd.closing && fd.sysref == 0 && fd.sysfd != syscall.InvalidHandle {
		// Poller may want to unregister fd in readiness notification mechanism,
		// so this must be executed before closesocket.
		fd.pd.Close()
		closesocket(fd.sysfd)
		fd.sysfd = syscall.InvalidHandle
		// no need for a finalizer anymore
		runtime.SetFinalizer(fd, nil)
	}
	fd.sysmu.Unlock()
}

func (fd *netFD) Close() error {
	if err := fd.incref(true); err != nil {
		return err
	}
	defer fd.decref()
	// unblock pending reader and writer
	fd.pd.Evict()
	// wait for both reader and writer to exit
	fd.rio.Lock()
	defer fd.rio.Unlock()
	fd.wio.Lock()
	defer fd.wio.Unlock()
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

// Read from network.

type readOp struct {
	bufOp
}

func (o *readOp) Submit() error {
	var d, f uint32
	return syscall.WSARecv(syscall.Handle(o.fd.sysfd), &o.buf, 1, &d, &f, &o.o, nil)
}

func (o *readOp) Name() string {
	return "WSARecv"
}

func (fd *netFD) Read(buf []byte) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	fd.rio.Lock()
	defer fd.rio.Unlock()
	var o readOp
	o.Init(fd, buf, 'r')
	n, err := iosrv.ExecIO(&o)
	if err == nil && n == 0 {
		err = io.EOF
	}
	return n, err
}

// ReadFrom from network.

type readFromOp struct {
	bufOp
	rsa  syscall.RawSockaddrAny
	rsan int32
}

func (o *readFromOp) Submit() error {
	var d, f uint32
	return syscall.WSARecvFrom(o.fd.sysfd, &o.buf, 1, &d, &f, &o.rsa, &o.rsan, &o.o, nil)
}

func (o *readFromOp) Name() string {
	return "WSARecvFrom"
}

func (fd *netFD) ReadFrom(buf []byte) (n int, sa syscall.Sockaddr, err error) {
	if len(buf) == 0 {
		return 0, nil, nil
	}
	if err := fd.incref(false); err != nil {
		return 0, nil, err
	}
	defer fd.decref()
	fd.rio.Lock()
	defer fd.rio.Unlock()
	var o readFromOp
	o.Init(fd, buf, 'r')
	o.rsan = int32(unsafe.Sizeof(o.rsa))
	n, err = iosrv.ExecIO(&o)
	if err != nil {
		return 0, nil, err
	}
	sa, _ = o.rsa.Sockaddr()
	return
}

// Write to network.

type writeOp struct {
	bufOp
}

func (o *writeOp) Submit() error {
	var d uint32
	return syscall.WSASend(o.fd.sysfd, &o.buf, 1, &d, 0, &o.o, nil)
}

func (o *writeOp) Name() string {
	return "WSASend"
}

func (fd *netFD) Write(buf []byte) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	fd.wio.Lock()
	defer fd.wio.Unlock()
	var o writeOp
	o.Init(fd, buf, 'w')
	return iosrv.ExecIO(&o)
}

// WriteTo to network.

type writeToOp struct {
	bufOp
	sa syscall.Sockaddr
}

func (o *writeToOp) Submit() error {
	var d uint32
	return syscall.WSASendto(o.fd.sysfd, &o.buf, 1, &d, 0, o.sa, &o.o, nil)
}

func (o *writeToOp) Name() string {
	return "WSASendto"
}

func (fd *netFD) WriteTo(buf []byte, sa syscall.Sockaddr) (int, error) {
	if len(buf) == 0 {
		return 0, nil
	}
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	fd.wio.Lock()
	defer fd.wio.Unlock()
	var o writeToOp
	o.Init(fd, buf, 'w')
	o.sa = sa
	return iosrv.ExecIO(&o)
}

// Accept new network connections.

type acceptOp struct {
	anOp
	newsock syscall.Handle
	attrs   [2]syscall.RawSockaddrAny // space for local and remote address only
}

func (o *acceptOp) Submit() error {
	var d uint32
	l := uint32(unsafe.Sizeof(o.attrs[0]))
	return syscall.AcceptEx(o.fd.sysfd, o.newsock,
		(*byte)(unsafe.Pointer(&o.attrs[0])), 0, l, l, &d, &o.o)
}

func (o *acceptOp) Name() string {
	return "AcceptEx"
}

func (fd *netFD) accept(toAddr func(syscall.Sockaddr) Addr) (*netFD, error) {
	if err := fd.incref(false); err != nil {
		return nil, err
	}
	defer fd.decref()

	// Get new socket.
	s, err := sysSocket(fd.family, fd.sotype, 0)
	if err != nil {
		return nil, &OpError{"socket", fd.net, fd.laddr, err}
	}

	// Associate our new socket with IOCP.
	netfd, err := newFD(s, fd.family, fd.sotype, fd.net)
	if err != nil {
		closesocket(s)
		return nil, &OpError{"accept", fd.net, fd.laddr, err}
	}

	// Submit accept request.
	fd.rio.Lock()
	defer fd.rio.Unlock()
	var o acceptOp
	o.Init(fd, 'r')
	o.newsock = s
	_, err = iosrv.ExecIO(&o)
	if err != nil {
		netfd.Close()
		return nil, err
	}

	// Inherit properties of the listening socket.
	err = syscall.Setsockopt(s, syscall.SOL_SOCKET, syscall.SO_UPDATE_ACCEPT_CONTEXT, (*byte)(unsafe.Pointer(&fd.sysfd)), int32(unsafe.Sizeof(fd.sysfd)))
	if err != nil {
		netfd.Close()
		return nil, &OpError{"Setsockopt", fd.net, fd.laddr, err}
	}

	// Get local and peer addr out of AcceptEx buffer.
	var lrsa, rrsa *syscall.RawSockaddrAny
	var llen, rlen int32
	l := uint32(unsafe.Sizeof(*lrsa))
	syscall.GetAcceptExSockaddrs((*byte)(unsafe.Pointer(&o.attrs[0])),
		0, l, l, &lrsa, &llen, &rrsa, &rlen)
	lsa, _ := lrsa.Sockaddr()
	rsa, _ := rrsa.Sockaddr()

	netfd.setAddr(toAddr(lsa), toAddr(rsa))
	return netfd, nil
}

// Unimplemented functions.

func (fd *netFD) dup() (*os.File, error) {
	// TODO: Implement this
	return nil, os.NewSyscallError("dup", syscall.EWINDOWS)
}

var errNoSupport = errors.New("address family not supported")

func (fd *netFD) ReadMsg(p []byte, oob []byte) (n, oobn, flags int, sa syscall.Sockaddr, err error) {
	return 0, 0, 0, nil, errNoSupport
}

func (fd *netFD) WriteMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	return 0, 0, errNoSupport
}

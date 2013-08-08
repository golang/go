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

var (
	canCancelIO                               bool // determines if CancelIoEx API is present
	skipSyncNotif                             bool
	hasLoadSetFileCompletionNotificationModes bool
)

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

	hasLoadSetFileCompletionNotificationModes = syscall.LoadSetFileCompletionNotificationModes() == nil
	if hasLoadSetFileCompletionNotificationModes {
		// It's not safe to use FILE_SKIP_COMPLETION_PORT_ON_SUCCESS if non IFS providers are installed:
		// http://support.microsoft.com/kb/2568167
		skipSyncNotif = true
		protos := [2]int32{syscall.IPPROTO_TCP, 0}
		var buf [32]syscall.WSAProtocolInfo
		len := uint32(unsafe.Sizeof(buf))
		n, err := syscall.WSAEnumProtocols(&protos[0], &buf[0], &len)
		if err != nil {
			skipSyncNotif = false
		} else {
			for i := int32(0); i < n; i++ {
				if buf[i].ServiceFlags1&syscall.XP1_IFS_HANDLES == 0 {
					skipSyncNotif = false
					break
				}
			}
		}
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

// operation contains superset of data necessary to perform all async IO.
type operation struct {
	// Used by IOCP interface, it must be first field
	// of the struct, as our code rely on it.
	o syscall.Overlapped

	// fields used by runtime.netpoll
	runtimeCtx uintptr
	mode       int32
	errno      int32
	qty        uint32

	// fields used only by net package
	mu     sync.Mutex
	fd     *netFD
	errc   chan error
	buf    syscall.WSABuf
	sa     syscall.Sockaddr
	rsa    *syscall.RawSockaddrAny
	rsan   int32
	handle syscall.Handle
	flags  uint32
}

func (o *operation) InitBuf(buf []byte) {
	o.buf.Len = uint32(len(buf))
	o.buf.Buf = nil
	if len(buf) != 0 {
		o.buf.Buf = (*byte)(unsafe.Pointer(&buf[0]))
	}
}

// ioSrv executes net IO requests.
type ioSrv struct {
	req chan ioSrvReq
}

type ioSrvReq struct {
	o      *operation
	submit func(o *operation) error // if nil, cancel the operation
}

// ProcessRemoteIO will execute submit IO requests on behalf
// of other goroutines, all on a single os thread, so it can
// cancel them later. Results of all operations will be sent
// back to their requesters via channel supplied in request.
// It is used only when the CancelIoEx API is unavailable.
func (s *ioSrv) ProcessRemoteIO() {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	for r := range s.req {
		if r.submit != nil {
			r.o.errc <- r.submit(r.o)
		} else {
			r.o.errc <- syscall.CancelIo(r.o.fd.sysfd)
		}
	}
}

// ExecIO executes a single IO operation o. It submits and cancels
// IO in the current thread for systems where Windows CancelIoEx API
// is available. Alternatively, it passes the request onto
// runtime netpoll and waits for completion or cancels request.
func (s *ioSrv) ExecIO(o *operation, name string, submit func(o *operation) error) (int, error) {
	fd := o.fd
	// Notify runtime netpoll about starting IO.
	err := fd.pd.Prepare(int(o.mode))
	if err != nil {
		return 0, &OpError{name, fd.net, fd.laddr, err}
	}
	// Start IO.
	if canCancelIO {
		err = submit(o)
	} else {
		// Send request to a special dedicated thread,
		// so it can stop the IO with CancelIO later.
		s.req <- ioSrvReq{o, submit}
		err = <-o.errc
	}
	switch err {
	case nil:
		// IO completed immediately
		if o.fd.skipSyncNotif {
			// No completion message will follow, so return immediately.
			return int(o.qty), nil
		}
		// Need to get our completion message anyway.
	case syscall.ERROR_IO_PENDING:
		// IO started, and we have to wait for its completion.
		err = nil
	default:
		return 0, &OpError{name, fd.net, fd.laddr, err}
	}
	// Wait for our request to complete.
	err = fd.pd.Wait(int(o.mode))
	if err == nil {
		// All is good. Extract our IO results and return.
		if o.errno != 0 {
			err = syscall.Errno(o.errno)
			return 0, &OpError{name, fd.net, fd.laddr, err}
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
		err := syscall.CancelIoEx(fd.sysfd, &o.o)
		// Assuming ERROR_NOT_FOUND is returned, if IO is completed.
		if err != nil && err != syscall.ERROR_NOT_FOUND {
			// TODO(brainman): maybe do something else, but panic.
			panic(err)
		}
	} else {
		s.req <- ioSrvReq{o, nil}
		<-o.errc
	}
	// Wait for cancellation to complete.
	fd.pd.WaitCanceled(int(o.mode))
	if o.errno != 0 {
		err = syscall.Errno(o.errno)
		if err == syscall.ERROR_OPERATION_ABORTED { // IO Canceled
			err = netpollErr
		}
		return 0, &OpError{name, fd.net, fd.laddr, err}
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
		iosrv.req = make(chan ioSrvReq)
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
	sysfd         syscall.Handle
	family        int
	sotype        int
	isConnected   bool
	skipSyncNotif bool
	net           string
	laddr         Addr
	raddr         Addr

	rop operation // read operation
	wop operation // write operation

	// wait server
	pd pollDesc
}

func newFD(sysfd syscall.Handle, family, sotype int, net string) (*netFD, error) {
	if initErr != nil {
		return nil, initErr
	}
	onceStartServer.Do(startServer)
	return &netFD{sysfd: sysfd, family: family, sotype: sotype, net: net}, nil
}

func (fd *netFD) init() error {
	if err := fd.pd.Init(fd); err != nil {
		return err
	}
	if hasLoadSetFileCompletionNotificationModes {
		// We do not use events, so we can skip them always.
		flags := uint8(syscall.FILE_SKIP_SET_EVENT_ON_HANDLE)
		// It's not safe to skip completion notifications for UDP:
		// http://blogs.technet.com/b/winserverperformance/archive/2008/06/26/designing-applications-for-high-performance-part-iii.aspx
		if skipSyncNotif && fd.net == "tcp" {
			flags |= syscall.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS
		}
		err := syscall.SetFileCompletionNotificationModes(fd.sysfd, flags)
		if err == nil && flags&syscall.FILE_SKIP_COMPLETION_PORT_ON_SUCCESS != 0 {
			fd.skipSyncNotif = true
		}
	}
	fd.rop.mode = 'r'
	fd.wop.mode = 'w'
	fd.rop.fd = fd
	fd.wop.fd = fd
	fd.rop.runtimeCtx = fd.pd.runtimeCtx
	fd.wop.runtimeCtx = fd.pd.runtimeCtx
	if !canCancelIO {
		fd.rop.errc = make(chan error)
		fd.wop.errc = make(chan error)
	}
	return nil
}

func (fd *netFD) setAddr(laddr, raddr Addr) {
	fd.laddr = laddr
	fd.raddr = raddr
	runtime.SetFinalizer(fd, (*netFD).Close)
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
	o := &fd.wop
	o.mu.Lock()
	defer o.mu.Unlock()
	o.sa = ra
	_, err := iosrv.ExecIO(o, "ConnectEx", func(o *operation) error {
		return syscall.ConnectEx(o.fd.sysfd, o.sa, nil, 0, nil, &o.o)
	})
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
	fd.rop.mu.Lock()
	fd.wop.mu.Lock()
	fd.rop.mu.Unlock()
	fd.wop.mu.Unlock()
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

func (fd *netFD) Read(buf []byte) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	o := &fd.rop
	o.mu.Lock()
	defer o.mu.Unlock()
	o.InitBuf(buf)
	n, err := iosrv.ExecIO(o, "WSARecv", func(o *operation) error {
		return syscall.WSARecv(o.fd.sysfd, &o.buf, 1, &o.qty, &o.flags, &o.o, nil)
	})
	if err == nil && n == 0 {
		err = io.EOF
	}
	return n, err
}

func (fd *netFD) ReadFrom(buf []byte) (n int, sa syscall.Sockaddr, err error) {
	if len(buf) == 0 {
		return 0, nil, nil
	}
	if err := fd.incref(false); err != nil {
		return 0, nil, err
	}
	defer fd.decref()
	o := &fd.rop
	o.mu.Lock()
	defer o.mu.Unlock()
	o.InitBuf(buf)
	n, err = iosrv.ExecIO(o, "WSARecvFrom", func(o *operation) error {
		if o.rsa == nil {
			o.rsa = new(syscall.RawSockaddrAny)
		}
		o.rsan = int32(unsafe.Sizeof(*o.rsa))
		return syscall.WSARecvFrom(o.fd.sysfd, &o.buf, 1, &o.qty, &o.flags, o.rsa, &o.rsan, &o.o, nil)
	})
	if err != nil {
		return 0, nil, err
	}
	sa, _ = o.rsa.Sockaddr()
	return
}

func (fd *netFD) Write(buf []byte) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	o := &fd.wop
	o.mu.Lock()
	defer o.mu.Unlock()
	o.InitBuf(buf)
	return iosrv.ExecIO(o, "WSASend", func(o *operation) error {
		return syscall.WSASend(o.fd.sysfd, &o.buf, 1, &o.qty, 0, &o.o, nil)
	})
}

func (fd *netFD) WriteTo(buf []byte, sa syscall.Sockaddr) (int, error) {
	if len(buf) == 0 {
		return 0, nil
	}
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	o := &fd.wop
	o.mu.Lock()
	defer o.mu.Unlock()
	o.InitBuf(buf)
	o.sa = sa
	return iosrv.ExecIO(o, "WSASendto", func(o *operation) error {
		return syscall.WSASendto(o.fd.sysfd, &o.buf, 1, &o.qty, 0, o.sa, &o.o, nil)
	})
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
	if err := netfd.init(); err != nil {
		fd.Close()
		return nil, err
	}

	// Submit accept request.
	o := &fd.rop
	o.mu.Lock()
	defer o.mu.Unlock()
	o.handle = s
	var rawsa [2]syscall.RawSockaddrAny
	o.rsan = int32(unsafe.Sizeof(rawsa[0]))
	_, err = iosrv.ExecIO(o, "AcceptEx", func(o *operation) error {
		return syscall.AcceptEx(o.fd.sysfd, o.handle, (*byte)(unsafe.Pointer(&rawsa[0])), 0, uint32(o.rsan), uint32(o.rsan), &o.qty, &o.o)
	})
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
	syscall.GetAcceptExSockaddrs((*byte)(unsafe.Pointer(&rawsa[0])),
		0, uint32(o.rsan), uint32(o.rsan), &lrsa, &llen, &rrsa, &rlen)
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

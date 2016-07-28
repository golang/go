// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package net

import (
	"context"
	"io"
	"os"
	"runtime"
	"sync/atomic"
	"syscall"
)

// Network file descriptor.
type netFD struct {
	// locking/lifetime of sysfd + serialize access to Read and Write methods
	fdmu fdMutex

	// immutable until Close
	sysfd       int
	family      int
	sotype      int
	isConnected bool
	net         string
	laddr       Addr
	raddr       Addr

	// wait server
	pd pollDesc
}

func sysInit() {
}

func newFD(sysfd, family, sotype int, net string) (*netFD, error) {
	return &netFD{sysfd: sysfd, family: family, sotype: sotype, net: net}, nil
}

func (fd *netFD) init() error {
	if err := fd.pd.init(fd); err != nil {
		return err
	}
	return nil
}

func (fd *netFD) setAddr(laddr, raddr Addr) {
	fd.laddr = laddr
	fd.raddr = raddr
	runtime.SetFinalizer(fd, (*netFD).Close)
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

func (fd *netFD) connect(ctx context.Context, la, ra syscall.Sockaddr) (ret error) {
	// Do not need to call fd.writeLock here,
	// because fd is not yet accessible to user,
	// so no concurrent operations are possible.
	switch err := connectFunc(fd.sysfd, ra); err {
	case syscall.EINPROGRESS, syscall.EALREADY, syscall.EINTR:
	case nil, syscall.EISCONN:
		select {
		case <-ctx.Done():
			return mapErr(ctx.Err())
		default:
		}
		if err := fd.init(); err != nil {
			return err
		}
		return nil
	case syscall.EINVAL:
		// On Solaris we can see EINVAL if the socket has
		// already been accepted and closed by the server.
		// Treat this as a successful connection--writes to
		// the socket will see EOF.  For details and a test
		// case in C see https://golang.org/issue/6828.
		if runtime.GOOS == "solaris" {
			return nil
		}
		fallthrough
	default:
		return os.NewSyscallError("connect", err)
	}
	if err := fd.init(); err != nil {
		return err
	}
	if deadline, _ := ctx.Deadline(); !deadline.IsZero() {
		fd.setWriteDeadline(deadline)
		defer fd.setWriteDeadline(noDeadline)
	}

	// Start the "interrupter" goroutine, if this context might be canceled.
	// (The background context cannot)
	//
	// The interrupter goroutine waits for the context to be done and
	// interrupts the dial (by altering the fd's write deadline, which
	// wakes up waitWrite).
	if ctx != context.Background() {
		// Wait for the interrupter goroutine to exit before returning
		// from connect.
		done := make(chan struct{})
		interruptRes := make(chan error)
		defer func() {
			close(done)
			if ctxErr := <-interruptRes; ctxErr != nil && ret == nil {
				// The interrupter goroutine called setWriteDeadline,
				// but the connect code below had returned from
				// waitWrite already and did a successful connect (ret
				// == nil). Because we've now poisoned the connection
				// by making it unwritable, don't return a successful
				// dial. This was issue 16523.
				ret = ctxErr
				fd.Close() // prevent a leak
			}
		}()
		go func() {
			select {
			case <-ctx.Done():
				// Force the runtime's poller to immediately give up
				// waiting for writability, unblocking waitWrite
				// below.
				fd.setWriteDeadline(aLongTimeAgo)
				testHookCanceledDial()
				interruptRes <- ctx.Err()
			case <-done:
				interruptRes <- nil
			}
		}()
	}

	for {
		// Performing multiple connect system calls on a
		// non-blocking socket under Unix variants does not
		// necessarily result in earlier errors being
		// returned. Instead, once runtime-integrated network
		// poller tells us that the socket is ready, get the
		// SO_ERROR socket option to see if the connection
		// succeeded or failed. See issue 7474 for further
		// details.
		if err := fd.pd.waitWrite(); err != nil {
			select {
			case <-ctx.Done():
				return mapErr(ctx.Err())
			default:
			}
			return err
		}
		nerr, err := getsockoptIntFunc(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_ERROR)
		if err != nil {
			return os.NewSyscallError("getsockopt", err)
		}
		switch err := syscall.Errno(nerr); err {
		case syscall.EINPROGRESS, syscall.EALREADY, syscall.EINTR:
		case syscall.Errno(0), syscall.EISCONN:
			if runtime.GOOS != "darwin" {
				return nil
			}
			// See golang.org/issue/14548.
			// On Darwin, multiple connect system calls on
			// a non-blocking socket never harm SO_ERROR.
			switch err := connectFunc(fd.sysfd, ra); err {
			case nil, syscall.EISCONN:
				return nil
			}
		default:
			return os.NewSyscallError("getsockopt", err)
		}
	}
}

func (fd *netFD) destroy() {
	// Poller may want to unregister fd in readiness notification mechanism,
	// so this must be executed before closeFunc.
	fd.pd.close()
	closeFunc(fd.sysfd)
	fd.sysfd = -1
	runtime.SetFinalizer(fd, nil)
}

func (fd *netFD) Close() error {
	if !fd.fdmu.increfAndClose() {
		return errClosing
	}
	// Unblock any I/O.  Once it all unblocks and returns,
	// so that it cannot be referring to fd.sysfd anymore,
	// the final decref will close fd.sysfd. This should happen
	// fairly quickly, since all the I/O is non-blocking, and any
	// attempts to block in the pollDesc will return errClosing.
	fd.pd.evict()
	fd.decref()
	return nil
}

func (fd *netFD) shutdown(how int) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("shutdown", syscall.Shutdown(fd.sysfd, how))
}

func (fd *netFD) closeRead() error {
	return fd.shutdown(syscall.SHUT_RD)
}

func (fd *netFD) closeWrite() error {
	return fd.shutdown(syscall.SHUT_WR)
}

func (fd *netFD) Read(p []byte) (n int, err error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if len(p) == 0 {
		// If the caller wanted a zero byte read, return immediately
		// without trying. (But after acquiring the readLock.) Otherwise
		// syscall.Read returns 0, nil and eofError turns that into
		// io.EOF.
		// TODO(bradfitz): make it wait for readability? (Issue 15735)
		return 0, nil
	}
	if err := fd.pd.prepareRead(); err != nil {
		return 0, err
	}
	for {
		n, err = syscall.Read(fd.sysfd, p)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN {
				if err = fd.pd.waitRead(); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		break
	}
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError("read", err)
	}
	return
}

func (fd *netFD) readFrom(p []byte) (n int, sa syscall.Sockaddr, err error) {
	if err := fd.readLock(); err != nil {
		return 0, nil, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(); err != nil {
		return 0, nil, err
	}
	for {
		n, sa, err = syscall.Recvfrom(fd.sysfd, p, 0)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN {
				if err = fd.pd.waitRead(); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		break
	}
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError("recvfrom", err)
	}
	return
}

func (fd *netFD) readMsg(p []byte, oob []byte) (n, oobn, flags int, sa syscall.Sockaddr, err error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, nil, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(); err != nil {
		return 0, 0, 0, nil, err
	}
	for {
		n, oobn, flags, sa, err = syscall.Recvmsg(fd.sysfd, p, oob, 0)
		if err != nil {
			// TODO(dfc) should n and oobn be set to 0
			if err == syscall.EAGAIN {
				if err = fd.pd.waitRead(); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		break
	}
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError("recvmsg", err)
	}
	return
}

func (fd *netFD) Write(p []byte) (nn int, err error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(); err != nil {
		return 0, err
	}
	for {
		var n int
		n, err = syscall.Write(fd.sysfd, p[nn:])
		if n > 0 {
			nn += n
		}
		if nn == len(p) {
			break
		}
		if err == syscall.EAGAIN {
			if err = fd.pd.waitWrite(); err == nil {
				continue
			}
		}
		if err != nil {
			break
		}
		if n == 0 {
			err = io.ErrUnexpectedEOF
			break
		}
	}
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError("write", err)
	}
	return nn, err
}

func (fd *netFD) writeTo(p []byte, sa syscall.Sockaddr) (n int, err error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(); err != nil {
		return 0, err
	}
	for {
		err = syscall.Sendto(fd.sysfd, p, 0, sa)
		if err == syscall.EAGAIN {
			if err = fd.pd.waitWrite(); err == nil {
				continue
			}
		}
		break
	}
	if err == nil {
		n = len(p)
	}
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError("sendto", err)
	}
	return
}

func (fd *netFD) writeMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(); err != nil {
		return 0, 0, err
	}
	for {
		n, err = syscall.SendmsgN(fd.sysfd, p, oob, sa, 0)
		if err == syscall.EAGAIN {
			if err = fd.pd.waitWrite(); err == nil {
				continue
			}
		}
		break
	}
	if err == nil {
		oobn = len(oob)
	}
	if _, ok := err.(syscall.Errno); ok {
		err = os.NewSyscallError("sendmsg", err)
	}
	return
}

func (fd *netFD) accept() (netfd *netFD, err error) {
	if err := fd.readLock(); err != nil {
		return nil, err
	}
	defer fd.readUnlock()

	var s int
	var rsa syscall.Sockaddr
	if err = fd.pd.prepareRead(); err != nil {
		return nil, err
	}
	for {
		s, rsa, err = accept(fd.sysfd)
		if err != nil {
			nerr, ok := err.(*os.SyscallError)
			if !ok {
				return nil, err
			}
			switch nerr.Err {
			case syscall.EAGAIN:
				if err = fd.pd.waitRead(); err == nil {
					continue
				}
			case syscall.ECONNABORTED:
				// This means that a socket on the
				// listen queue was closed before we
				// Accept()ed it; it's a silly error,
				// so try again.
				continue
			}
			return nil, err
		}
		break
	}

	if netfd, err = newFD(s, fd.family, fd.sotype, fd.net); err != nil {
		closeFunc(s)
		return nil, err
	}
	if err = netfd.init(); err != nil {
		fd.Close()
		return nil, err
	}
	lsa, _ := syscall.Getsockname(netfd.sysfd)
	netfd.setAddr(netfd.addrFunc()(lsa), netfd.addrFunc()(rsa))
	return netfd, nil
}

// tryDupCloexec indicates whether F_DUPFD_CLOEXEC should be used.
// If the kernel doesn't support it, this is set to 0.
var tryDupCloexec = int32(1)

func dupCloseOnExec(fd int) (newfd int, err error) {
	if atomic.LoadInt32(&tryDupCloexec) == 1 {
		r0, _, e1 := syscall.Syscall(syscall.SYS_FCNTL, uintptr(fd), syscall.F_DUPFD_CLOEXEC, 0)
		if runtime.GOOS == "darwin" && e1 == syscall.EBADF {
			// On OS X 10.6 and below (but we only support
			// >= 10.6), F_DUPFD_CLOEXEC is unsupported
			// and fcntl there falls back (undocumented)
			// to doing an ioctl instead, returning EBADF
			// in this case because fd is not of the
			// expected device fd type. Treat it as
			// EINVAL instead, so we fall back to the
			// normal dup path.
			// TODO: only do this on 10.6 if we can detect 10.6
			// cheaply.
			e1 = syscall.EINVAL
		}
		switch e1 {
		case 0:
			return int(r0), nil
		case syscall.EINVAL:
			// Old kernel. Fall back to the portable way
			// from now on.
			atomic.StoreInt32(&tryDupCloexec, 0)
		default:
			return -1, os.NewSyscallError("fcntl", e1)
		}
	}
	return dupCloseOnExecOld(fd)
}

// dupCloseOnExecUnixOld is the traditional way to dup an fd and
// set its O_CLOEXEC bit, using two system calls.
func dupCloseOnExecOld(fd int) (newfd int, err error) {
	syscall.ForkLock.RLock()
	defer syscall.ForkLock.RUnlock()
	newfd, err = syscall.Dup(fd)
	if err != nil {
		return -1, os.NewSyscallError("dup", err)
	}
	syscall.CloseOnExec(newfd)
	return
}

func (fd *netFD) dup() (f *os.File, err error) {
	ns, err := dupCloseOnExec(fd.sysfd)
	if err != nil {
		return nil, err
	}

	// We want blocking mode for the new fd, hence the double negative.
	// This also puts the old fd into blocking mode, meaning that
	// I/O will block the thread instead of letting us use the epoll server.
	// Everything will still work, just with more threads.
	if err = syscall.SetNonblock(ns, false); err != nil {
		return nil, os.NewSyscallError("setnonblock", err)
	}

	return os.NewFile(uintptr(ns), fd.name()), nil
}

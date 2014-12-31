// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package net

import (
	"io"
	"os"
	"runtime"
	"sync/atomic"
	"syscall"
	"time"
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

func dial(network string, ra Addr, dialer func(time.Time) (Conn, error), deadline time.Time) (Conn, error) {
	return dialer(deadline)
}

func newFD(sysfd, family, sotype int, net string) (*netFD, error) {
	return &netFD{sysfd: sysfd, family: family, sotype: sotype, net: net}, nil
}

func (fd *netFD) init() error {
	if err := fd.pd.Init(fd); err != nil {
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

func (fd *netFD) connect(la, ra syscall.Sockaddr, deadline time.Time) error {
	// Do not need to call fd.writeLock here,
	// because fd is not yet accessible to user,
	// so no concurrent operations are possible.
	switch err := syscall.Connect(fd.sysfd, ra); err {
	case syscall.EINPROGRESS, syscall.EALREADY, syscall.EINTR:
	case nil, syscall.EISCONN:
		if !deadline.IsZero() && deadline.Before(time.Now()) {
			return errTimeout
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
		// case in C see http://golang.org/issue/6828.
		if runtime.GOOS == "solaris" {
			return nil
		}
		fallthrough
	default:
		return err
	}
	if err := fd.init(); err != nil {
		return err
	}
	if !deadline.IsZero() {
		fd.setWriteDeadline(deadline)
		defer fd.setWriteDeadline(noDeadline)
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
		if err := fd.pd.WaitWrite(); err != nil {
			return err
		}
		nerr, err := syscall.GetsockoptInt(fd.sysfd, syscall.SOL_SOCKET, syscall.SO_ERROR)
		if err != nil {
			return err
		}
		switch err := syscall.Errno(nerr); err {
		case syscall.EINPROGRESS, syscall.EALREADY, syscall.EINTR:
		case syscall.Errno(0), syscall.EISCONN:
			return nil
		default:
			return err
		}
	}
}

func (fd *netFD) destroy() {
	// Poller may want to unregister fd in readiness notification mechanism,
	// so this must be executed before closesocket.
	fd.pd.Close()
	closesocket(fd.sysfd)
	fd.sysfd = -1
	runtime.SetFinalizer(fd, nil)
}

// Add a reference to this fd.
// Returns an error if the fd cannot be used.
func (fd *netFD) incref() error {
	if !fd.fdmu.Incref() {
		return errClosing
	}
	return nil
}

// Remove a reference to this FD and close if we've been asked to do so
// (and there are no references left).
func (fd *netFD) decref() {
	if fd.fdmu.Decref() {
		fd.destroy()
	}
}

// Add a reference to this fd and lock for reading.
// Returns an error if the fd cannot be used.
func (fd *netFD) readLock() error {
	if !fd.fdmu.RWLock(true) {
		return errClosing
	}
	return nil
}

// Unlock for reading and remove a reference to this FD.
func (fd *netFD) readUnlock() {
	if fd.fdmu.RWUnlock(true) {
		fd.destroy()
	}
}

// Add a reference to this fd and lock for writing.
// Returns an error if the fd cannot be used.
func (fd *netFD) writeLock() error {
	if !fd.fdmu.RWLock(false) {
		return errClosing
	}
	return nil
}

// Unlock for writing and remove a reference to this FD.
func (fd *netFD) writeUnlock() {
	if fd.fdmu.RWUnlock(false) {
		fd.destroy()
	}
}

func (fd *netFD) Close() error {
	fd.pd.Lock() // needed for both fd.incref(true) and pollDesc.Evict
	if !fd.fdmu.IncrefAndClose() {
		fd.pd.Unlock()
		return errClosing
	}
	// Unblock any I/O.  Once it all unblocks and returns,
	// so that it cannot be referring to fd.sysfd anymore,
	// the final decref will close fd.sysfd.  This should happen
	// fairly quickly, since all the I/O is non-blocking, and any
	// attempts to block in the pollDesc will return errClosing.
	doWakeup := fd.pd.Evict()
	fd.pd.Unlock()
	fd.decref()
	if doWakeup {
		fd.pd.Wakeup()
	}
	return nil
}

func (fd *netFD) shutdown(how int) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.Shutdown(fd.sysfd, how)
	if err != nil {
		return &OpError{"shutdown", fd.net, fd.laddr, err}
	}
	return nil
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
	if err := fd.pd.PrepareRead(); err != nil {
		return 0, &OpError{"read", fd.net, fd.raddr, err}
	}
	for {
		n, err = syscall.Read(int(fd.sysfd), p)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN {
				if err = fd.pd.WaitRead(); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		break
	}
	if err != nil && err != io.EOF {
		err = &OpError{"read", fd.net, fd.raddr, err}
	}
	return
}

func (fd *netFD) readFrom(p []byte) (n int, sa syscall.Sockaddr, err error) {
	if err := fd.readLock(); err != nil {
		return 0, nil, err
	}
	defer fd.readUnlock()
	if err := fd.pd.PrepareRead(); err != nil {
		return 0, nil, &OpError{"read", fd.net, fd.laddr, err}
	}
	for {
		n, sa, err = syscall.Recvfrom(fd.sysfd, p, 0)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN {
				if err = fd.pd.WaitRead(); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		break
	}
	if err != nil && err != io.EOF {
		err = &OpError{"read", fd.net, fd.laddr, err}
	}
	return
}

func (fd *netFD) readMsg(p []byte, oob []byte) (n, oobn, flags int, sa syscall.Sockaddr, err error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, nil, err
	}
	defer fd.readUnlock()
	if err := fd.pd.PrepareRead(); err != nil {
		return 0, 0, 0, nil, &OpError{"read", fd.net, fd.laddr, err}
	}
	for {
		n, oobn, flags, sa, err = syscall.Recvmsg(fd.sysfd, p, oob, 0)
		if err != nil {
			// TODO(dfc) should n and oobn be set to 0
			if err == syscall.EAGAIN {
				if err = fd.pd.WaitRead(); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		break
	}
	if err != nil && err != io.EOF {
		err = &OpError{"read", fd.net, fd.laddr, err}
	}
	return
}

func (fd *netFD) Write(p []byte) (nn int, err error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.PrepareWrite(); err != nil {
		return 0, &OpError{"write", fd.net, fd.raddr, err}
	}
	for {
		var n int
		n, err = syscall.Write(int(fd.sysfd), p[nn:])
		if n > 0 {
			nn += n
		}
		if nn == len(p) {
			break
		}
		if err == syscall.EAGAIN {
			if err = fd.pd.WaitWrite(); err == nil {
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

func (fd *netFD) writeTo(p []byte, sa syscall.Sockaddr) (n int, err error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.PrepareWrite(); err != nil {
		return 0, &OpError{"write", fd.net, fd.raddr, err}
	}
	for {
		err = syscall.Sendto(fd.sysfd, p, 0, sa)
		if err == syscall.EAGAIN {
			if err = fd.pd.WaitWrite(); err == nil {
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

func (fd *netFD) writeMsg(p []byte, oob []byte, sa syscall.Sockaddr) (n int, oobn int, err error) {
	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.PrepareWrite(); err != nil {
		return 0, 0, &OpError{"write", fd.net, fd.raddr, err}
	}
	for {
		n, err = syscall.SendmsgN(fd.sysfd, p, oob, sa, 0)
		if err == syscall.EAGAIN {
			if err = fd.pd.WaitWrite(); err == nil {
				continue
			}
		}
		break
	}
	if err == nil {
		oobn = len(oob)
	} else {
		err = &OpError{"write", fd.net, fd.raddr, err}
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
	if err = fd.pd.PrepareRead(); err != nil {
		return nil, &OpError{"accept", fd.net, fd.laddr, err}
	}
	for {
		s, rsa, err = accept(fd.sysfd)
		if err != nil {
			if err == syscall.EAGAIN {
				if err = fd.pd.WaitRead(); err == nil {
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
			// expected device fd type.  Treat it as
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
			return -1, e1
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
		return -1, err
	}
	syscall.CloseOnExec(newfd)
	return
}

func (fd *netFD) dup() (f *os.File, err error) {
	ns, err := dupCloseOnExec(fd.sysfd)
	if err != nil {
		return nil, &OpError{"dup", fd.net, fd.laddr, err}
	}

	// We want blocking mode for the new fd, hence the double negative.
	// This also puts the old fd into blocking mode, meaning that
	// I/O will block the thread instead of letting us use the epoll server.
	// Everything will still work, just with more threads.
	if err = syscall.SetNonblock(ns, false); err != nil {
		return nil, &OpError{"setnonblock", fd.net, fd.laddr, err}
	}

	return os.NewFile(uintptr(ns), fd.name()), nil
}

func closesocket(s int) error {
	return syscall.Close(s)
}

func skipRawSocketTests() (skip bool, skipmsg string, err error) {
	if os.Getuid() != 0 {
		return true, "skipping test; must be root", nil
	}
	return false, "", nil
}

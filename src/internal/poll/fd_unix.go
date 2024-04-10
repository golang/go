// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package poll

import (
	"internal/itoa"
	"internal/syscall/unix"
	"io"
	"sync/atomic"
	"syscall"
)

// FD is a file descriptor. The net and os packages use this type as a
// field of a larger type representing a network connection or OS file.
type FD struct {
	// Lock sysfd and serialize access to Read and Write methods.
	fdmu fdMutex

	// System file descriptor. Immutable until Close.
	Sysfd int

	// Platform dependent state of the file descriptor.
	SysFile

	// I/O poller.
	pd pollDesc

	// Semaphore signaled when file is closed.
	csema uint32

	// Non-zero if this file has been set to blocking mode.
	isBlocking uint32

	// Whether this is a streaming descriptor, as opposed to a
	// packet-based descriptor like a UDP socket. Immutable.
	IsStream bool

	// Whether a zero byte read indicates EOF. This is false for a
	// message based socket connection.
	ZeroReadIsEOF bool

	// Whether this is a file rather than a network socket.
	isFile bool
}

// Init initializes the FD. The Sysfd field should already be set.
// This can be called multiple times on a single FD.
// The net argument is a network name from the net package (e.g., "tcp"),
// or "file".
// Set pollable to true if fd should be managed by runtime netpoll.
func (fd *FD) Init(net string, pollable bool) error {
	fd.SysFile.init()

	// We don't actually care about the various network types.
	if net == "file" {
		fd.isFile = true
	}
	if !pollable {
		fd.isBlocking = 1
		return nil
	}
	err := fd.pd.init(fd)
	if err != nil {
		// If we could not initialize the runtime poller,
		// assume we are using blocking mode.
		fd.isBlocking = 1
	}
	return err
}

// Destroy closes the file descriptor. This is called when there are
// no remaining references.
func (fd *FD) destroy() error {
	// Poller may want to unregister fd in readiness notification mechanism,
	// so this must be executed before CloseFunc.
	fd.pd.close()

	err := fd.SysFile.destroy(fd.Sysfd)

	fd.Sysfd = -1
	runtime_Semrelease(&fd.csema)
	return err
}

// Close closes the FD. The underlying file descriptor is closed by the
// destroy method when there are no remaining references.
func (fd *FD) Close() error {
	if !fd.fdmu.increfAndClose() {
		return errClosing(fd.isFile)
	}

	// Unblock any I/O.  Once it all unblocks and returns,
	// so that it cannot be referring to fd.sysfd anymore,
	// the final decref will close fd.sysfd. This should happen
	// fairly quickly, since all the I/O is non-blocking, and any
	// attempts to block in the pollDesc will return errClosing(fd.isFile).
	fd.pd.evict()

	// The call to decref will call destroy if there are no other
	// references.
	err := fd.decref()

	// Wait until the descriptor is closed. If this was the only
	// reference, it is already closed. Only wait if the file has
	// not been set to blocking mode, as otherwise any current I/O
	// may be blocking, and that would block the Close.
	// No need for an atomic read of isBlocking, increfAndClose means
	// we have exclusive access to fd.
	if fd.isBlocking == 0 {
		runtime_Semacquire(&fd.csema)
	}

	return err
}

// SetBlocking puts the file into blocking mode.
func (fd *FD) SetBlocking() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	// Atomic store so that concurrent calls to SetBlocking
	// do not cause a race condition. isBlocking only ever goes
	// from 0 to 1 so there is no real race here.
	atomic.StoreUint32(&fd.isBlocking, 1)
	return syscall.SetNonblock(fd.Sysfd, false)
}

// Darwin and FreeBSD can't read or write 2GB+ files at a time,
// even on 64-bit systems.
// The same is true of socket implementations on many systems.
// See golang.org/issue/7812 and golang.org/issue/16266.
// Use 1GB instead of, say, 2GB-1, to keep subsequent reads aligned.
const maxRW = 1 << 30

// Read implements io.Reader.
func (fd *FD) Read(p []byte) (int, error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if len(p) == 0 {
		// If the caller wanted a zero byte read, return immediately
		// without trying (but after acquiring the readLock).
		// Otherwise syscall.Read returns 0, nil which looks like
		// io.EOF.
		// TODO(bradfitz): make it wait for readability? (Issue 15735)
		return 0, nil
	}
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, err
	}
	if fd.IsStream && len(p) > maxRW {
		p = p[:maxRW]
	}
	for {
		n, err := ignoringEINTRIO(syscall.Read, fd.Sysfd, p)
		if err != nil {
			n = 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, err
	}
}

// Pread wraps the pread system call.
func (fd *FD) Pread(p []byte, off int64) (int, error) {
	// Call incref, not readLock, because since pread specifies the
	// offset it is independent from other reads.
	// Similarly, using the poller doesn't make sense for pread.
	if err := fd.incref(); err != nil {
		return 0, err
	}
	if fd.IsStream && len(p) > maxRW {
		p = p[:maxRW]
	}
	var (
		n   int
		err error
	)
	for {
		n, err = syscall.Pread(fd.Sysfd, p, off)
		if err != syscall.EINTR {
			break
		}
	}
	if err != nil {
		n = 0
	}
	fd.decref()
	err = fd.eofError(n, err)
	return n, err
}

// ReadFrom wraps the recvfrom network call.
func (fd *FD) ReadFrom(p []byte) (int, syscall.Sockaddr, error) {
	if err := fd.readLock(); err != nil {
		return 0, nil, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, nil, err
	}
	for {
		n, sa, err := syscall.Recvfrom(fd.Sysfd, p, 0)
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			n = 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, sa, err
	}
}

// ReadFromInet4 wraps the recvfrom network call for IPv4.
func (fd *FD) ReadFromInet4(p []byte, from *syscall.SockaddrInet4) (int, error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, err
	}
	for {
		n, err := unix.RecvfromInet4(fd.Sysfd, p, 0, from)
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			n = 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, err
	}
}

// ReadFromInet6 wraps the recvfrom network call for IPv6.
func (fd *FD) ReadFromInet6(p []byte, from *syscall.SockaddrInet6) (int, error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, err
	}
	for {
		n, err := unix.RecvfromInet6(fd.Sysfd, p, 0, from)
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			n = 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, err
	}
}

// ReadMsg wraps the recvmsg network call.
func (fd *FD) ReadMsg(p []byte, oob []byte, flags int) (int, int, int, syscall.Sockaddr, error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, nil, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, 0, 0, nil, err
	}
	for {
		n, oobn, sysflags, sa, err := syscall.Recvmsg(fd.Sysfd, p, oob, flags)
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			// TODO(dfc) should n and oobn be set to 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, oobn, sysflags, sa, err
	}
}

// ReadMsgInet4 is ReadMsg, but specialized for syscall.SockaddrInet4.
func (fd *FD) ReadMsgInet4(p []byte, oob []byte, flags int, sa4 *syscall.SockaddrInet4) (int, int, int, error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, 0, 0, err
	}
	for {
		n, oobn, sysflags, err := unix.RecvmsgInet4(fd.Sysfd, p, oob, flags, sa4)
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			// TODO(dfc) should n and oobn be set to 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, oobn, sysflags, err
	}
}

// ReadMsgInet6 is ReadMsg, but specialized for syscall.SockaddrInet6.
func (fd *FD) ReadMsgInet6(p []byte, oob []byte, flags int, sa6 *syscall.SockaddrInet6) (int, int, int, error) {
	if err := fd.readLock(); err != nil {
		return 0, 0, 0, err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return 0, 0, 0, err
	}
	for {
		n, oobn, sysflags, err := unix.RecvmsgInet6(fd.Sysfd, p, oob, flags, sa6)
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			// TODO(dfc) should n and oobn be set to 0
			if err == syscall.EAGAIN && fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		}
		err = fd.eofError(n, err)
		return n, oobn, sysflags, err
	}
}

// Write implements io.Writer.
func (fd *FD) Write(p []byte) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, err
	}
	var nn int
	for {
		max := len(p)
		if fd.IsStream && max-nn > maxRW {
			max = nn + maxRW
		}
		n, err := ignoringEINTRIO(syscall.Write, fd.Sysfd, p[nn:max])
		if n > 0 {
			if n > max-nn {
				// This can reportedly happen when using
				// some VPN software. Issue #61060.
				// If we don't check this we will panic
				// with slice bounds out of range.
				// Use a more informative panic.
				panic("invalid return from write: got " + itoa.Itoa(n) + " from a write of " + itoa.Itoa(max-nn))
			}
			nn += n
		}
		if nn == len(p) {
			return nn, err
		}
		if err == syscall.EAGAIN && fd.pd.pollable() {
			if err = fd.pd.waitWrite(fd.isFile); err == nil {
				continue
			}
		}
		if err != nil {
			return nn, err
		}
		if n == 0 {
			return nn, io.ErrUnexpectedEOF
		}
	}
}

// Pwrite wraps the pwrite system call.
func (fd *FD) Pwrite(p []byte, off int64) (int, error) {
	// Call incref, not writeLock, because since pwrite specifies the
	// offset it is independent from other writes.
	// Similarly, using the poller doesn't make sense for pwrite.
	if err := fd.incref(); err != nil {
		return 0, err
	}
	defer fd.decref()
	var nn int
	for {
		max := len(p)
		if fd.IsStream && max-nn > maxRW {
			max = nn + maxRW
		}
		n, err := syscall.Pwrite(fd.Sysfd, p[nn:max], off+int64(nn))
		if err == syscall.EINTR {
			continue
		}
		if n > 0 {
			nn += n
		}
		if nn == len(p) {
			return nn, err
		}
		if err != nil {
			return nn, err
		}
		if n == 0 {
			return nn, io.ErrUnexpectedEOF
		}
	}
}

// WriteToInet4 wraps the sendto network call for IPv4 addresses.
func (fd *FD) WriteToInet4(p []byte, sa *syscall.SockaddrInet4) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, err
	}
	for {
		err := unix.SendtoInet4(fd.Sysfd, p, 0, sa)
		if err == syscall.EINTR {
			continue
		}
		if err == syscall.EAGAIN && fd.pd.pollable() {
			if err = fd.pd.waitWrite(fd.isFile); err == nil {
				continue
			}
		}
		if err != nil {
			return 0, err
		}
		return len(p), nil
	}
}

// WriteToInet6 wraps the sendto network call for IPv6 addresses.
func (fd *FD) WriteToInet6(p []byte, sa *syscall.SockaddrInet6) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, err
	}
	for {
		err := unix.SendtoInet6(fd.Sysfd, p, 0, sa)
		if err == syscall.EINTR {
			continue
		}
		if err == syscall.EAGAIN && fd.pd.pollable() {
			if err = fd.pd.waitWrite(fd.isFile); err == nil {
				continue
			}
		}
		if err != nil {
			return 0, err
		}
		return len(p), nil
	}
}

// WriteTo wraps the sendto network call.
func (fd *FD) WriteTo(p []byte, sa syscall.Sockaddr) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, err
	}
	for {
		err := syscall.Sendto(fd.Sysfd, p, 0, sa)
		if err == syscall.EINTR {
			continue
		}
		if err == syscall.EAGAIN && fd.pd.pollable() {
			if err = fd.pd.waitWrite(fd.isFile); err == nil {
				continue
			}
		}
		if err != nil {
			return 0, err
		}
		return len(p), nil
	}
}

// WriteMsg wraps the sendmsg network call.
func (fd *FD) WriteMsg(p []byte, oob []byte, sa syscall.Sockaddr) (int, int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, 0, err
	}
	for {
		n, err := syscall.SendmsgN(fd.Sysfd, p, oob, sa, 0)
		if err == syscall.EINTR {
			continue
		}
		if err == syscall.EAGAIN && fd.pd.pollable() {
			if err = fd.pd.waitWrite(fd.isFile); err == nil {
				continue
			}
		}
		if err != nil {
			return n, 0, err
		}
		return n, len(oob), err
	}
}

// WriteMsgInet4 is WriteMsg specialized for syscall.SockaddrInet4.
func (fd *FD) WriteMsgInet4(p []byte, oob []byte, sa *syscall.SockaddrInet4) (int, int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, 0, err
	}
	for {
		n, err := unix.SendmsgNInet4(fd.Sysfd, p, oob, sa, 0)
		if err == syscall.EINTR {
			continue
		}
		if err == syscall.EAGAIN && fd.pd.pollable() {
			if err = fd.pd.waitWrite(fd.isFile); err == nil {
				continue
			}
		}
		if err != nil {
			return n, 0, err
		}
		return n, len(oob), err
	}
}

// WriteMsgInet6 is WriteMsg specialized for syscall.SockaddrInet6.
func (fd *FD) WriteMsgInet6(p []byte, oob []byte, sa *syscall.SockaddrInet6) (int, int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, 0, err
	}
	for {
		n, err := unix.SendmsgNInet6(fd.Sysfd, p, oob, sa, 0)
		if err == syscall.EINTR {
			continue
		}
		if err == syscall.EAGAIN && fd.pd.pollable() {
			if err = fd.pd.waitWrite(fd.isFile); err == nil {
				continue
			}
		}
		if err != nil {
			return n, 0, err
		}
		return n, len(oob), err
	}
}

// Accept wraps the accept network call.
func (fd *FD) Accept() (int, syscall.Sockaddr, string, error) {
	if err := fd.readLock(); err != nil {
		return -1, nil, "", err
	}
	defer fd.readUnlock()

	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return -1, nil, "", err
	}
	for {
		s, rsa, errcall, err := accept(fd.Sysfd)
		if err == nil {
			return s, rsa, "", err
		}
		switch err {
		case syscall.EINTR:
			continue
		case syscall.EAGAIN:
			if fd.pd.pollable() {
				if err = fd.pd.waitRead(fd.isFile); err == nil {
					continue
				}
			}
		case syscall.ECONNABORTED:
			// This means that a socket on the listen
			// queue was closed before we Accept()ed it;
			// it's a silly error, so try again.
			continue
		}
		return -1, nil, errcall, err
	}
}

// Fchmod wraps syscall.Fchmod.
func (fd *FD) Fchmod(mode uint32) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return ignoringEINTR(func() error {
		return syscall.Fchmod(fd.Sysfd, mode)
	})
}

// Fstat wraps syscall.Fstat
func (fd *FD) Fstat(s *syscall.Stat_t) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return ignoringEINTR(func() error {
		return syscall.Fstat(fd.Sysfd, s)
	})
}

// dupCloexecUnsupported indicates whether F_DUPFD_CLOEXEC is supported by the kernel.
var dupCloexecUnsupported atomic.Bool

// DupCloseOnExec dups fd and marks it close-on-exec.
func DupCloseOnExec(fd int) (int, string, error) {
	if syscall.F_DUPFD_CLOEXEC != 0 && !dupCloexecUnsupported.Load() {
		r0, err := unix.Fcntl(fd, syscall.F_DUPFD_CLOEXEC, 0)
		if err == nil {
			return r0, "", nil
		}
		switch err {
		case syscall.EINVAL, syscall.ENOSYS:
			// Old kernel, or js/wasm (which returns
			// ENOSYS). Fall back to the portable way from
			// now on.
			dupCloexecUnsupported.Store(true)
		default:
			return -1, "fcntl", err
		}
	}
	return dupCloseOnExecOld(fd)
}

// Dup duplicates the file descriptor.
func (fd *FD) Dup() (int, string, error) {
	if err := fd.incref(); err != nil {
		return -1, "", err
	}
	defer fd.decref()
	return DupCloseOnExec(fd.Sysfd)
}

// On Unix variants only, expose the IO event for the net code.

// WaitWrite waits until data can be read from fd.
func (fd *FD) WaitWrite() error {
	return fd.pd.waitWrite(fd.isFile)
}

// WriteOnce is for testing only. It makes a single write call.
func (fd *FD) WriteOnce(p []byte) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	return ignoringEINTRIO(syscall.Write, fd.Sysfd, p)
}

// RawRead invokes the user-defined function f for a read operation.
func (fd *FD) RawRead(f func(uintptr) bool) error {
	if err := fd.readLock(); err != nil {
		return err
	}
	defer fd.readUnlock()
	if err := fd.pd.prepareRead(fd.isFile); err != nil {
		return err
	}
	for {
		if f(uintptr(fd.Sysfd)) {
			return nil
		}
		if err := fd.pd.waitRead(fd.isFile); err != nil {
			return err
		}
	}
}

// RawWrite invokes the user-defined function f for a write operation.
func (fd *FD) RawWrite(f func(uintptr) bool) error {
	if err := fd.writeLock(); err != nil {
		return err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return err
	}
	for {
		if f(uintptr(fd.Sysfd)) {
			return nil
		}
		if err := fd.pd.waitWrite(fd.isFile); err != nil {
			return err
		}
	}
}

// ignoringEINTRIO is like ignoringEINTR, but just for IO calls.
func ignoringEINTRIO(fn func(fd int, p []byte) (int, error), fd int, p []byte) (int, error) {
	for {
		n, err := fn(fd, p)
		if err != syscall.EINTR {
			return n, err
		}
	}
}

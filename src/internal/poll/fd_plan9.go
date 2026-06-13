// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"errors"
	"internal/stringslite"
	"io"
	"sync"
	"syscall"
	"time"
)

type FD struct {
	// Lock sysfd and serialize access to Read and Write methods.
	fdmu fdMutex

	Destroy func()

	// deadlines
	rmu       sync.Mutex
	wmu       sync.Mutex
	raio      *asyncIO
	waio      *asyncIO
	rtimer    *time.Timer
	wtimer    *time.Timer
	rtimedout bool      // set true when read deadline has been reached
	wtimedout bool      // set true when write deadline has been reached
	rdeadline time.Time // absolute read deadline (zero if none)
	wdeadline time.Time // absolute write deadline (zero if none)

	// Whether this is a normal file.
	// On Plan 9 we do not use this package for ordinary files,
	// so this is always false, but the field is present because
	// shared code in fd_mutex.go checks it.
	isFile bool
}

// We need this to close out a file descriptor when it is unlocked,
// but the real implementation has to live in the net package because
// it uses os.File's.
func (fd *FD) destroy() error {
	if fd.Destroy != nil {
		fd.Destroy()
	}
	return nil
}

// Close handles the locking for closing an FD. The real operation
// is in the net package.
func (fd *FD) Close() error {
	if !fd.fdmu.increfAndClose() {
		return errClosing(fd.isFile)
	}
	return nil
}

// Read implements io.Reader.
func (fd *FD) Read(fn func([]byte) (int, error), b []byte) (int, error) {
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if len(b) == 0 {
		return 0, nil
	}
	fd.rmu.Lock()
	if fd.rtimedout {
		fd.rmu.Unlock()
		return 0, ErrDeadlineExceeded
	}
	fd.raio = newAsyncIO(fn, b)
	fd.rmu.Unlock()
	n, err := fd.raio.Wait()
	fd.rmu.Lock()
	fd.raio = nil
	timedOut := fd.rtimedout
	deadline := fd.rdeadline
	fd.rmu.Unlock()
	if isHangup(err) {
		err = io.EOF
	}
	if isInterrupted(err) {
		err = ErrDeadlineExceeded
	}
	// If the deadline has expired by wall-clock or the timer flagged
	// rtimedout, prefer the deadline error over a "soft" result (no error
	// or io.EOF with 0 bytes). This covers two races on Plan 9:
	//
	//   1. The deadline timer fired (rtimedout=true) but the Cancel note
	//      lost to the underlying syscall completing naturally.
	//   2. The syscall returned a spurious EOF on a fresh loopback TCP
	//      connection under SMP before the timer goroutine got a chance
	//      to set rtimedout=true; the wall clock is already past the
	//      deadline, so reporting timeout matches what the caller asked
	//      for.
	//
	// We only override n==0 results; data observed before the deadline is
	// still useful and we don't want to discard it.
	if n == 0 && (err == nil || err == io.EOF) {
		if timedOut || (!deadline.IsZero() && !time.Now().Before(deadline)) {
			err = ErrDeadlineExceeded
		}
	}
	// Defense in depth for very short deadlines on plan9 SMP: the kernel
	// can return a spurious (0, EOF) on a fresh loopback TCP connection
	// before either the timer goroutine has updated rtimedout or the wall
	// clock has crossed the deadline.  If we still have a (0, EOF) here
	// but the deadline is "soon" (within 1ms), spin briefly until the
	// deadline elapses and the timer fires; then re-check.  We cap the
	// extra wait so a connection that legitimately observes a fast EOF on
	// a long-deadline connection isn't held up.
	if n == 0 && err == io.EOF && !deadline.IsZero() {
		if remaining := time.Until(deadline); remaining > 0 && remaining <= time.Millisecond {
			time.Sleep(remaining)
			fd.rmu.Lock()
			timedOut = fd.rtimedout
			fd.rmu.Unlock()
			if timedOut || !time.Now().Before(deadline) {
				err = ErrDeadlineExceeded
			}
		}
	}
	return n, err
}

// Write implements io.Writer.
func (fd *FD) Write(fn func([]byte) (int, error), b []byte) (int, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	fd.wmu.Lock()
	if fd.wtimedout {
		fd.wmu.Unlock()
		return 0, ErrDeadlineExceeded
	}
	fd.waio = newAsyncIO(fn, b)
	fd.wmu.Unlock()
	n, err := fd.waio.Wait()
	fd.wmu.Lock()
	fd.waio = nil
	timedOut := fd.wtimedout
	deadline := fd.wdeadline
	fd.wmu.Unlock()
	if isInterrupted(err) {
		err = ErrDeadlineExceeded
	}
	// Symmetric to Read: if the write deadline fired before fn returned and
	// we got an ambiguous "wrote nothing" result, prefer the deadline error.
	if n == 0 && err == nil {
		if timedOut || (!deadline.IsZero() && !time.Now().Before(deadline)) {
			err = ErrDeadlineExceeded
		}
	}
	return n, err
}

// SetDeadline sets the read and write deadlines associated with fd.
func (fd *FD) SetDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'r'+'w')
}

// SetReadDeadline sets the read deadline associated with fd.
func (fd *FD) SetReadDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'r')
}

// SetWriteDeadline sets the write deadline associated with fd.
func (fd *FD) SetWriteDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'w')
}

func setDeadlineImpl(fd *FD, t time.Time, mode int) error {
	d := t.Sub(time.Now())
	if mode == 'r' || mode == 'r'+'w' {
		fd.rmu.Lock()
		defer fd.rmu.Unlock()
		if fd.rtimer != nil {
			fd.rtimer.Stop()
			fd.rtimer = nil
		}
		fd.rtimedout = false
		fd.rdeadline = t
	}
	if mode == 'w' || mode == 'r'+'w' {
		fd.wmu.Lock()
		defer fd.wmu.Unlock()
		if fd.wtimer != nil {
			fd.wtimer.Stop()
			fd.wtimer = nil
		}
		fd.wtimedout = false
		fd.wdeadline = t
	}
	if !t.IsZero() && d > 0 {
		// Interrupt I/O operation once timer has expired
		if mode == 'r' || mode == 'r'+'w' {
			var timer *time.Timer
			timer = time.AfterFunc(d, func() {
				fd.rmu.Lock()
				defer fd.rmu.Unlock()
				if fd.rtimer != timer {
					// deadline was changed
					return
				}
				fd.rtimedout = true
				if fd.raio != nil {
					fd.raio.Cancel()
				}
			})
			fd.rtimer = timer
		}
		if mode == 'w' || mode == 'r'+'w' {
			var timer *time.Timer
			timer = time.AfterFunc(d, func() {
				fd.wmu.Lock()
				defer fd.wmu.Unlock()
				if fd.wtimer != timer {
					// deadline was changed
					return
				}
				fd.wtimedout = true
				if fd.waio != nil {
					fd.waio.Cancel()
				}
			})
			fd.wtimer = timer
		}
	}
	if !t.IsZero() && d <= 0 {
		// Interrupt current I/O operation
		if mode == 'r' || mode == 'r'+'w' {
			fd.rtimedout = true
			if fd.raio != nil {
				fd.raio.Cancel()
			}
		}
		if mode == 'w' || mode == 'r'+'w' {
			fd.wtimedout = true
			if fd.waio != nil {
				fd.waio.Cancel()
			}
		}
	}
	return nil
}

// On Plan 9 only, expose the locking for the net code.

// ReadLock wraps FD.readLock.
func (fd *FD) ReadLock() error {
	return fd.readLock()
}

// ReadUnlock wraps FD.readUnlock.
func (fd *FD) ReadUnlock() {
	fd.readUnlock()
}

func isHangup(err error) bool {
	return err != nil && stringslite.HasSuffix(err.Error(), "Hangup")
}

func isInterrupted(err error) bool {
	return err != nil && stringslite.HasSuffix(err.Error(), "interrupted")
}

// IsPollDescriptor reports whether fd is the descriptor being used by the poller.
// This is only used for testing.
func IsPollDescriptor(fd uintptr) bool {
	return false
}

// RawControl invokes the user-defined function f for a non-IO
// operation.
func (fd *FD) RawControl(f func(uintptr)) error {
	return errors.New("not implemented")
}

// RawRead invokes the user-defined function f for a read operation.
func (fd *FD) RawRead(f func(uintptr) bool) error {
	return errors.New("not implemented")
}

// RawWrite invokes the user-defined function f for a write operation.
func (fd *FD) RawWrite(f func(uintptr) bool) error {
	return errors.New("not implemented")
}

func DupCloseOnExec(fd int) (int, string, error) {
	nfd, err := syscall.Dup(int(fd), -1)
	if err != nil {
		return 0, "dup", err
	}
	// Plan9 has no syscall.CloseOnExec but
	// its forkAndExecInChild closes all fds
	// not related to the fork+exec.
	return nfd, "", nil
}

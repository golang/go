// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"errors"
	"io"
	"sync/atomic"
	"time"
)

type atomicBool int32

func (b *atomicBool) isSet() bool { return atomic.LoadInt32((*int32)(b)) != 0 }
func (b *atomicBool) setFalse()   { atomic.StoreInt32((*int32)(b), 0) }
func (b *atomicBool) setTrue()    { atomic.StoreInt32((*int32)(b), 1) }

type FD struct {
	// Lock sysfd and serialize access to Read and Write methods.
	fdmu fdMutex

	Destroy func()

	// deadlines
	raio      *asyncIO
	waio      *asyncIO
	rtimer    *time.Timer
	wtimer    *time.Timer
	rtimedout atomicBool // set true when read deadline has been reached
	wtimedout atomicBool // set true when write deadline has been reached

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
	if fd.rtimedout.isSet() {
		return 0, ErrTimeout
	}
	if err := fd.readLock(); err != nil {
		return 0, err
	}
	defer fd.readUnlock()
	if len(b) == 0 {
		return 0, nil
	}
	fd.raio = newAsyncIO(fn, b)
	n, err := fd.raio.Wait()
	fd.raio = nil
	if isHangup(err) {
		err = io.EOF
	}
	if isInterrupted(err) {
		err = ErrTimeout
	}
	return n, err
}

// Write implements io.Writer.
func (fd *FD) Write(fn func([]byte) (int, error), b []byte) (int, error) {
	if fd.wtimedout.isSet() {
		return 0, ErrTimeout
	}
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	fd.waio = newAsyncIO(fn, b)
	n, err := fd.waio.Wait()
	fd.waio = nil
	if isInterrupted(err) {
		err = ErrTimeout
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
		fd.rtimedout.setFalse()
	}
	if mode == 'w' || mode == 'r'+'w' {
		fd.wtimedout.setFalse()
	}
	if t.IsZero() || d < 0 {
		// Stop timer
		if mode == 'r' || mode == 'r'+'w' {
			if fd.rtimer != nil {
				fd.rtimer.Stop()
			}
			fd.rtimer = nil
		}
		if mode == 'w' || mode == 'r'+'w' {
			if fd.wtimer != nil {
				fd.wtimer.Stop()
			}
			fd.wtimer = nil
		}
	} else {
		// Interrupt I/O operation once timer has expired
		if mode == 'r' || mode == 'r'+'w' {
			fd.rtimer = time.AfterFunc(d, func() {
				fd.rtimedout.setTrue()
				if fd.raio != nil {
					fd.raio.Cancel()
				}
			})
		}
		if mode == 'w' || mode == 'r'+'w' {
			fd.wtimer = time.AfterFunc(d, func() {
				fd.wtimedout.setTrue()
				if fd.waio != nil {
					fd.waio.Cancel()
				}
			})
		}
	}
	if !t.IsZero() && d < 0 {
		// Interrupt current I/O operation
		if mode == 'r' || mode == 'r'+'w' {
			fd.rtimedout.setTrue()
			if fd.raio != nil {
				fd.raio.Cancel()
			}
		}
		if mode == 'w' || mode == 'r'+'w' {
			fd.wtimedout.setTrue()
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
	return err != nil && stringsHasSuffix(err.Error(), "Hangup")
}

func isInterrupted(err error) bool {
	return err != nil && stringsHasSuffix(err.Error(), "interrupted")
}

// PollDescriptor returns the descriptor being used by the poller,
// or ^uintptr(0) if there isn't one. This is only used for testing.
func PollDescriptor() uintptr {
	return ^uintptr(0)
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

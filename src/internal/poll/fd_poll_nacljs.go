// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl js,wasm

package poll

import (
	"syscall"
	"time"
)

type pollDesc struct {
	fd      *FD
	closing bool
}

func (pd *pollDesc) init(fd *FD) error { pd.fd = fd; return nil }

func (pd *pollDesc) close() {}

func (pd *pollDesc) evict() {
	pd.closing = true
	if pd.fd != nil {
		syscall.StopIO(pd.fd.Sysfd)
	}
}

func (pd *pollDesc) prepare(mode int, isFile bool) error {
	if pd.closing {
		return errClosing(isFile)
	}
	return nil
}

func (pd *pollDesc) prepareRead(isFile bool) error { return pd.prepare('r', isFile) }

func (pd *pollDesc) prepareWrite(isFile bool) error { return pd.prepare('w', isFile) }

func (pd *pollDesc) wait(mode int, isFile bool) error {
	if pd.closing {
		return errClosing(isFile)
	}
	if isFile { // TODO(neelance): wasm: Use callbacks from JS to block until the read/write finished.
		return nil
	}
	return ErrTimeout
}

func (pd *pollDesc) waitRead(isFile bool) error { return pd.wait('r', isFile) }

func (pd *pollDesc) waitWrite(isFile bool) error { return pd.wait('w', isFile) }

func (pd *pollDesc) waitCanceled(mode int) {}

func (pd *pollDesc) pollable() bool { return true }

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
	d := t.UnixNano()
	if t.IsZero() {
		d = 0
	}
	if err := fd.incref(); err != nil {
		return err
	}
	switch mode {
	case 'r':
		syscall.SetReadDeadline(fd.Sysfd, d)
	case 'w':
		syscall.SetWriteDeadline(fd.Sysfd, d)
	case 'r' + 'w':
		syscall.SetReadDeadline(fd.Sysfd, d)
		syscall.SetWriteDeadline(fd.Sysfd, d)
	}
	fd.decref()
	return nil
}

// PollDescriptor returns the descriptor being used by the poller,
// or ^uintptr(0) if there isn't one. This is only used for testing.
func PollDescriptor() uintptr {
	return ^uintptr(0)
}

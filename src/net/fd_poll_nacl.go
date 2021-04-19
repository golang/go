// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"syscall"
	"time"
)

type pollDesc struct {
	fd      *netFD
	closing bool
}

func (pd *pollDesc) init(fd *netFD) error { pd.fd = fd; return nil }

func (pd *pollDesc) close() {}

func (pd *pollDesc) evict() {
	pd.closing = true
	if pd.fd != nil {
		syscall.StopIO(pd.fd.sysfd)
		runtime.KeepAlive(pd.fd)
	}
}

func (pd *pollDesc) prepare(mode int) error {
	if pd.closing {
		return errClosing
	}
	return nil
}

func (pd *pollDesc) prepareRead() error { return pd.prepare('r') }

func (pd *pollDesc) prepareWrite() error { return pd.prepare('w') }

func (pd *pollDesc) wait(mode int) error {
	if pd.closing {
		return errClosing
	}
	return errTimeout
}

func (pd *pollDesc) waitRead() error { return pd.wait('r') }

func (pd *pollDesc) waitWrite() error { return pd.wait('w') }

func (pd *pollDesc) waitCanceled(mode int) {}

func (pd *pollDesc) waitCanceledRead() {}

func (pd *pollDesc) waitCanceledWrite() {}

func (fd *netFD) setDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'r'+'w')
}

func (fd *netFD) setReadDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'r')
}

func (fd *netFD) setWriteDeadline(t time.Time) error {
	return setDeadlineImpl(fd, t, 'w')
}

func setDeadlineImpl(fd *netFD, t time.Time, mode int) error {
	d := t.UnixNano()
	if t.IsZero() {
		d = 0
	}
	if err := fd.incref(); err != nil {
		return err
	}
	switch mode {
	case 'r':
		syscall.SetReadDeadline(fd.sysfd, d)
	case 'w':
		syscall.SetWriteDeadline(fd.sysfd, d)
	case 'r' + 'w':
		syscall.SetReadDeadline(fd.sysfd, d)
		syscall.SetWriteDeadline(fd.sysfd, d)
	}
	fd.decref()
	return nil
}

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"
	"time"
)

type pollDesc struct {
	fd      *netFD
	closing bool
}

func (pd *pollDesc) Init(fd *netFD) error { pd.fd = fd; return nil }

func (pd *pollDesc) Close() {}

func (pd *pollDesc) Lock() {}

func (pd *pollDesc) Unlock() {}

func (pd *pollDesc) Wakeup() {}

func (pd *pollDesc) Evict() bool {
	pd.closing = true
	if pd.fd != nil {
		syscall.StopIO(pd.fd.sysfd)
	}
	return false
}

func (pd *pollDesc) Prepare(mode int) error {
	if pd.closing {
		return errClosing
	}
	return nil
}

func (pd *pollDesc) PrepareRead() error { return pd.Prepare('r') }

func (pd *pollDesc) PrepareWrite() error { return pd.Prepare('w') }

func (pd *pollDesc) Wait(mode int) error {
	if pd.closing {
		return errClosing
	}
	return errTimeout
}

func (pd *pollDesc) WaitRead() error { return pd.Wait('r') }

func (pd *pollDesc) WaitWrite() error { return pd.Wait('w') }

func (pd *pollDesc) WaitCanceled(mode int) {}

func (pd *pollDesc) WaitCanceledRead() {}

func (pd *pollDesc) WaitCanceledWrite() {}

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

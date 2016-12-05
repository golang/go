// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd windows solaris

package net

import (
	"runtime"
	"sync"
	"syscall"
	"time"
)

// runtimeNano returns the current value of the runtime clock in nanoseconds.
func runtimeNano() int64

func runtime_pollServerInit()
func runtime_pollOpen(fd uintptr) (uintptr, int)
func runtime_pollClose(ctx uintptr)
func runtime_pollWait(ctx uintptr, mode int) int
func runtime_pollWaitCanceled(ctx uintptr, mode int) int
func runtime_pollReset(ctx uintptr, mode int) int
func runtime_pollSetDeadline(ctx uintptr, d int64, mode int)
func runtime_pollUnblock(ctx uintptr)

type pollDesc struct {
	runtimeCtx uintptr
}

var serverInit sync.Once

func (pd *pollDesc) init(fd *netFD) error {
	serverInit.Do(runtime_pollServerInit)
	ctx, errno := runtime_pollOpen(uintptr(fd.sysfd))
	runtime.KeepAlive(fd)
	if errno != 0 {
		return syscall.Errno(errno)
	}
	pd.runtimeCtx = ctx
	return nil
}

func (pd *pollDesc) close() {
	if pd.runtimeCtx == 0 {
		return
	}
	runtime_pollClose(pd.runtimeCtx)
	pd.runtimeCtx = 0
}

// Evict evicts fd from the pending list, unblocking any I/O running on fd.
func (pd *pollDesc) evict() {
	if pd.runtimeCtx == 0 {
		return
	}
	runtime_pollUnblock(pd.runtimeCtx)
}

func (pd *pollDesc) prepare(mode int) error {
	res := runtime_pollReset(pd.runtimeCtx, mode)
	return convertErr(res)
}

func (pd *pollDesc) prepareRead() error {
	return pd.prepare('r')
}

func (pd *pollDesc) prepareWrite() error {
	return pd.prepare('w')
}

func (pd *pollDesc) wait(mode int) error {
	res := runtime_pollWait(pd.runtimeCtx, mode)
	return convertErr(res)
}

func (pd *pollDesc) waitRead() error {
	return pd.wait('r')
}

func (pd *pollDesc) waitWrite() error {
	return pd.wait('w')
}

func (pd *pollDesc) waitCanceled(mode int) {
	runtime_pollWaitCanceled(pd.runtimeCtx, mode)
}

func (pd *pollDesc) waitCanceledRead() {
	pd.waitCanceled('r')
}

func (pd *pollDesc) waitCanceledWrite() {
	pd.waitCanceled('w')
}

func convertErr(res int) error {
	switch res {
	case 0:
		return nil
	case 1:
		return errClosing
	case 2:
		return errTimeout
	}
	println("unreachable: ", res)
	panic("unreachable")
}

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
	diff := int64(time.Until(t))
	d := runtimeNano() + diff
	if d <= 0 && diff > 0 {
		// If the user has a deadline in the future, but the delay calculation
		// overflows, then set the deadline to the maximum possible value.
		d = 1<<63 - 1
	}
	if t.IsZero() {
		d = 0
	}
	if err := fd.incref(); err != nil {
		return err
	}
	runtime_pollSetDeadline(fd.pd.runtimeCtx, d, mode)
	fd.decref()
	return nil
}

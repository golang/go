// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd windows

package net

import (
	"sync"
	"syscall"
	"time"
)

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

func (pd *pollDesc) Init(fd *netFD) error {
	serverInit.Do(runtime_pollServerInit)
	ctx, errno := runtime_pollOpen(uintptr(fd.sysfd))
	if errno != 0 {
		return syscall.Errno(errno)
	}
	pd.runtimeCtx = ctx
	return nil
}

func (pd *pollDesc) Close() {
	if pd.runtimeCtx == 0 {
		return
	}
	runtime_pollClose(pd.runtimeCtx)
	pd.runtimeCtx = 0
}

func (pd *pollDesc) Lock() {
}

func (pd *pollDesc) Unlock() {
}

func (pd *pollDesc) Wakeup() {
}

// Evict evicts fd from the pending list, unblocking any I/O running on fd.
// Return value is whether the pollServer should be woken up.
func (pd *pollDesc) Evict() bool {
	if pd.runtimeCtx == 0 {
		return false
	}
	runtime_pollUnblock(pd.runtimeCtx)
	return false
}

func (pd *pollDesc) Prepare(mode int) error {
	res := runtime_pollReset(pd.runtimeCtx, mode)
	return convertErr(res)
}

func (pd *pollDesc) PrepareRead() error {
	return pd.Prepare('r')
}

func (pd *pollDesc) PrepareWrite() error {
	return pd.Prepare('w')
}

func (pd *pollDesc) Wait(mode int) error {
	res := runtime_pollWait(pd.runtimeCtx, mode)
	return convertErr(res)
}

func (pd *pollDesc) WaitRead() error {
	return pd.Wait('r')
}

func (pd *pollDesc) WaitWrite() error {
	return pd.Wait('w')
}

func (pd *pollDesc) WaitCanceled(mode int) {
	runtime_pollWaitCanceled(pd.runtimeCtx, mode)
}

func (pd *pollDesc) WaitCanceledRead() {
	pd.WaitCanceled('r')
}

func (pd *pollDesc) WaitCanceledWrite() {
	pd.WaitCanceled('w')
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
	d := t.UnixNano()
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

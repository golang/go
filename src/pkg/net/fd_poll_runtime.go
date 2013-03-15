// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux

package net

import (
	"sync"
	"syscall"
	"time"
)

func runtime_pollServerInit()
func runtime_pollOpen(fd int) (uintptr, int)
func runtime_pollClose(ctx uintptr)
func runtime_pollWait(ctx uintptr, mode int) int
func runtime_pollReset(ctx uintptr, mode int) int
func runtime_pollSetDeadline(ctx uintptr, d int64, mode int)
func runtime_pollUnblock(ctx uintptr)

var canCancelIO = true // used for testing current package

type pollDesc struct {
	runtimeCtx uintptr
}

var serverInit sync.Once

func sysInit() {
}

func (pd *pollDesc) Init(fd *netFD) error {
	serverInit.Do(runtime_pollServerInit)
	ctx, errno := runtime_pollOpen(fd.sysfd)
	if errno != 0 {
		return syscall.Errno(errno)
	}
	pd.runtimeCtx = ctx
	return nil
}

func (pd *pollDesc) Close() {
	runtime_pollClose(pd.runtimeCtx)
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
	runtime_pollUnblock(pd.runtimeCtx)
	return false
}

func (pd *pollDesc) PrepareRead() error {
	res := runtime_pollReset(pd.runtimeCtx, 'r')
	return convertErr(res)
}

func (pd *pollDesc) PrepareWrite() error {
	res := runtime_pollReset(pd.runtimeCtx, 'w')
	return convertErr(res)
}

func (pd *pollDesc) WaitRead() error {
	res := runtime_pollWait(pd.runtimeCtx, 'r')
	return convertErr(res)
}

func (pd *pollDesc) WaitWrite() error {
	res := runtime_pollWait(pd.runtimeCtx, 'w')
	return convertErr(res)
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
	panic("unreachable")
}

func setReadDeadline(fd *netFD, t time.Time) error {
	return setDeadlineImpl(fd, t, 'r')
}

func setWriteDeadline(fd *netFD, t time.Time) error {
	return setDeadlineImpl(fd, t, 'w')
}

func setDeadline(fd *netFD, t time.Time) error {
	return setDeadlineImpl(fd, t, 'r'+'w')
}

func setDeadlineImpl(fd *netFD, t time.Time, mode int) error {
	d := t.UnixNano()
	if t.IsZero() {
		d = 0
	}
	if err := fd.incref(false); err != nil {
		return err
	}
	runtime_pollSetDeadline(fd.pd.runtimeCtx, d, mode)
	fd.decref()
	return nil
}

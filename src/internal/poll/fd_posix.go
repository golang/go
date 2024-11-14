// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1 || windows

package poll

import (
	"io"
	"syscall"
)

// eofError returns io.EOF when fd is available for reading end of
// file.
func (fd *FD) eofError(n int, err error) error {
	if n == 0 && err == nil && fd.ZeroReadIsEOF {
		return io.EOF
	}
	return err
}

// Shutdown wraps syscall.Shutdown.
func (fd *FD) Shutdown(how int) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return syscall.Shutdown(fd.Sysfd, how)
}

// Fchown wraps syscall.Fchown.
func (fd *FD) Fchown(uid, gid int) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return ignoringEINTR(func { syscall.Fchown(fd.Sysfd, uid, gid) })
}

// Ftruncate wraps syscall.Ftruncate.
func (fd *FD) Ftruncate(size int64) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return ignoringEINTR(func { syscall.Ftruncate(fd.Sysfd, size) })
}

// RawControl invokes the user-defined function f for a non-IO
// operation.
func (fd *FD) RawControl(f func(uintptr)) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	f(uintptr(fd.Sysfd))
	return nil
}

// ignoringEINTR makes a function call and repeats it if it returns
// an EINTR error. This appears to be required even though we install all
// signal handlers with SA_RESTART: see #22838, #38033, #38836, #40846.
// Also #20400 and #36644 are issues in which a signal handler is
// installed without setting SA_RESTART. None of these are the common case,
// but there are enough of them that it seems that we can't avoid
// an EINTR loop.
func ignoringEINTR(fn func() error) error {
	for {
		err := fn()
		if err != syscall.EINTR {
			return err
		}
	}
}

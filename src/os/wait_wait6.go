// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd

package os

import (
	"runtime"
	"syscall"
)

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and reports whether it has done so.
// It does not actually call p.Wait.
func (p *Process) blockUntilWaitable() (bool, error) {
	var errno syscall.Errno
	for {
		_, errno = wait6(_P_PID, p.Pid, syscall.WEXITED|syscall.WNOWAIT)
		if errno != syscall.EINTR {
			break
		}
	}
	runtime.KeepAlive(p)
	if errno == syscall.ENOSYS {
		return false, nil
	} else if errno != 0 {
		return false, NewSyscallError("wait6", errno)
	}
	return true, nil
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We used to use this code for Darwin, but according to issue #19314
// waitid returns if the process is stopped, even when using WEXITED.

//go:build linux

package os

import (
	"internal/syscall/unix"
	"runtime"
	"syscall"
)

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and reports whether it has done so.
// It does not actually call p.Wait.
func (p *Process) blockUntilWaitable() (bool, error) {
	var info unix.SiginfoChild
	err := ignoringEINTR(func() error {
		return unix.Waitid(unix.P_PID, p.Pid, &info, syscall.WEXITED|syscall.WNOWAIT, nil)
	})
	runtime.KeepAlive(p)
	if err != nil {
		// waitid has been available since Linux 2.6.9, but
		// reportedly is not available in Ubuntu on Windows.
		// See issue 16610.
		if err == syscall.ENOSYS {
			return false, nil
		}
		return false, NewSyscallError("waitid", err)
	}
	return true, nil
}

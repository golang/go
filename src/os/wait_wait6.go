// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd

package os

import (
	"runtime"
	"syscall"
)

const _P_PID = 0

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and reports whether it has done so.
// It does not actually call p.Wait.
func (p *Process) blockUntilWaitable() (bool, error) {
	var errno syscall.Errno
	for {
		// The arguments on 32-bit FreeBSD look like the following:
		// - freebsd32_wait6_args{ idtype, id1, id2, status, options, wrusage, info } or
		// - freebsd32_wait6_args{ idtype, pad, id1, id2, status, options, wrusage, info } when PAD64_REQUIRED=1 on ARM, MIPS or PowerPC
		if runtime.GOARCH == "386" {
			_, _, errno = syscall.Syscall9(syscall.SYS_WAIT6, _P_PID, uintptr(p.Pid), 0, 0, syscall.WEXITED|syscall.WNOWAIT, 0, 0, 0, 0)
		} else if runtime.GOARCH == "arm" {
			_, _, errno = syscall.Syscall9(syscall.SYS_WAIT6, _P_PID, 0, uintptr(p.Pid), 0, 0, syscall.WEXITED|syscall.WNOWAIT, 0, 0, 0)
		} else {
			_, _, errno = syscall.Syscall6(syscall.SYS_WAIT6, _P_PID, uintptr(p.Pid), 0, syscall.WEXITED|syscall.WNOWAIT, 0, 0)
		}
		if errno != syscall.EINTR {
			break
		}
	}
	runtime.KeepAlive(p)
	if errno != 0 {
		return false, NewSyscallError("wait6", errno)
	}
	return true, nil
}

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
// succeed immediately, and returns whether it has done so.
// It does not actually call p.Wait.
func (p *Process) blockUntilWaitable() (bool, error) {
	var errno syscall.Errno
	switch runtime.GOARCH {
	case "386", "arm":
		// The arguments on 32-bit FreeBSD look like the
		// following:
		// - freebsd32_wait6_args{ idtype, id1, id2, status, options, wrusage, info } or
		// - freebsd32_wait6_args{ idtype, pad, id1, id2, status, options, wrusage, info } when PAD64_REQUIRED=1 on MIPS or PowerPC
		_, _, errno = syscall.Syscall9(syscall.SYS_WAIT6, _P_PID, 0, uintptr(p.Pid), 0, syscall.WEXITED|syscall.WNOWAIT, 0, 0, 0, 0)
	default:
		_, _, errno = syscall.Syscall6(syscall.SYS_WAIT6, _P_PID, uintptr(p.Pid), 0, syscall.WEXITED|syscall.WNOWAIT, 0, 0)
	}
	if errno != 0 {
		// The wait6 system call is supported only on FreeBSD
		// 9.3 and above, so it may return an ENOSYS error.
		// Also the system call may return an ECHILD error
		// when the child process has not finished the
		// transformation using execve system call.
		// In both cases, we just leave the care of child
		// process to the following wait4 system call in
		// Process.wait.
		if errno == syscall.ENOSYS || errno == syscall.ECHILD {
			return false, nil
		}
		return false, NewSyscallError("wait6", errno)
	}
	return true, nil
}

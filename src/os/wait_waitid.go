// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We used to use this code for Darwin, but according to issue #19314
// waitid returns if the process is stopped, even when using WEXITED.

//go:build linux

package os

import (
	"runtime"
	"syscall"
	"unsafe"
)

const _P_PID = 1

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and reports whether it has done so.
// It does not actually call p.Wait.
func (p *Process) blockUntilWaitable() (bool, error) {
	// The waitid system call expects a pointer to a siginfo_t,
	// which is 128 bytes on all Linux systems.
	// On darwin/amd64, it requires 104 bytes.
	// We don't care about the values it returns.
	var siginfo [16]uint64
	psig := &siginfo[0]
	var e syscall.Errno
	for {
		_, _, e = syscall.Syscall6(syscall.SYS_WAITID, _P_PID, uintptr(p.Pid), uintptr(unsafe.Pointer(psig)), syscall.WEXITED|syscall.WNOWAIT, 0, 0)
		if e != syscall.EINTR {
			break
		}
	}
	runtime.KeepAlive(p)
	if e != 0 {
		// waitid has been available since Linux 2.6.9, but
		// reportedly is not available in Ubuntu on Windows.
		// See issue 16610.
		if e == syscall.ENOSYS {
			return false, nil
		}
		return false, NewSyscallError("waitid", e)
	}
	return true, nil
}

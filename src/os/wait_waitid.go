// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin linux

package os

import (
	"runtime"
	"syscall"
	"unsafe"
)

const _P_PID = 1

// blockUntilWaitable attempts to block until a call to p.Wait will
// succeed immediately, and returns whether it has done so.
// It does not actually call p.Wait.
func (p *Process) blockUntilWaitable() (bool, error) {
	// The waitid system call expects a pointer to a siginfo_t,
	// which is 128 bytes on all GNU/Linux systems.
	// On Darwin, it requires greater than or equal to 64 bytes
	// for darwin/{386,arm} and 104 bytes for darwin/amd64.
	// We don't care about the values it returns.
	var siginfo [128]byte
	psig := &siginfo[0]
	_, _, e := syscall.Syscall6(syscall.SYS_WAITID, _P_PID, uintptr(p.Pid), uintptr(unsafe.Pointer(psig)), syscall.WEXITED|syscall.WNOWAIT, 0, 0)
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

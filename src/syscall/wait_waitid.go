// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// We used to used this code for Darwin, but according to issue #19314
// waitid returns if the process is stopped, even when using WEXITED.

//go:build linux

package syscall

import (
	"unsafe"
)

const (
	_P_ALL = 0
	_P_PID = 1
)

// blockUntilWaitable attempts to block until a call to Wait4 will
// succeed immediately, and reports whether it has done so.
// It does not actually call Wait4.
func blockUntilWaitable(searchPID int) (int, error) {
	// The waitid system call expects a pointer to a siginfo_t,
	// which is 128 bytes on all Linux systems.
	// On darwin/amd64, it requires 104 bytes.
	// We don't care about the values it returns.
	var siginfo [16]uint64
	psig := &siginfo[0]
	idType := _P_ALL
	if searchPID != 0 {
		idType = _P_PID
	}
	for {
		if pid, _, err := Syscall6(SYS_WAITID, uintptr(idType), uintptr(searchPID), uintptr(unsafe.Pointer(psig)), WEXITED|WNOWAIT, 0, 0); err == ENOSYS {
			// waitid has been available since Linux 2.6.9, but
			// reportedly is not available in Ubuntu on Windows.
			// See issue 16610.
			return 0, nil
		} else if err != EINTR {
			return int(pid), err
		}
	}
}

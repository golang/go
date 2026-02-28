// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"syscall"
)

const is64bit = ^uint(0) >> 63 // 0 for 32-bit hosts, 1 for 64-bit ones.

// SiginfoChild is a struct filled in by Linux waitid syscall.
// In C, siginfo_t contains a union with multiple members;
// this struct corresponds to one used when Signo is SIGCHLD.
//
// NOTE fields are exported to be used by TestSiginfoChildLayout.
type SiginfoChild struct {
	Signo       int32
	siErrnoCode                // Two int32 fields, swapped on MIPS.
	_           [is64bit]int32 // Extra padding for 64-bit hosts only.

	// End of common part. Beginning of signal-specific part.

	Pid    int32
	Uid    uint32
	Status int32

	// Pad to 128 bytes.
	_ [128 - (6+is64bit)*4]byte
}

const (
	// Possible values for SiginfoChild.Code field.
	_CLD_EXITED    int32 = 1
	_CLD_KILLED          = 2
	_CLD_DUMPED          = 3
	_CLD_TRAPPED         = 4
	_CLD_STOPPED         = 5
	_CLD_CONTINUED       = 6

	// These are the same as in syscall/syscall_linux.go.
	core      = 0x80
	stopped   = 0x7f
	continued = 0xffff
)

// WaitStatus converts SiginfoChild, as filled in by the waitid syscall,
// to syscall.WaitStatus.
func (s *SiginfoChild) WaitStatus() (ws syscall.WaitStatus) {
	switch s.Code {
	case _CLD_EXITED:
		ws = syscall.WaitStatus(s.Status << 8)
	case _CLD_DUMPED:
		ws = syscall.WaitStatus(s.Status) | core
	case _CLD_KILLED:
		ws = syscall.WaitStatus(s.Status)
	case _CLD_TRAPPED, _CLD_STOPPED:
		ws = syscall.WaitStatus(s.Status<<8) | stopped
	case _CLD_CONTINUED:
		ws = continued
	}
	return
}

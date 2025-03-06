// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package runtime

import (
	"internal/runtime/atomic"
	"internal/runtime/syscall"
	"unsafe"
)

var prSetVMAUnsupported atomic.Bool

// setVMAName calls prctl(PR_SET_VMA, PR_SET_VMA_ANON_NAME, start, len, name)
func setVMAName(start unsafe.Pointer, length uintptr, name string) {
	if unsupported := prSetVMAUnsupported.Load(); unsupported {
		return
	}

	var sysName [80]byte
	n := copy(sysName[:], " Go: ")
	copy(sysName[n:79], name) // leave final byte zero

	_, _, err := syscall.Syscall6(syscall.SYS_PRCTL, syscall.PR_SET_VMA, syscall.PR_SET_VMA_ANON_NAME, uintptr(start), length, uintptr(unsafe.Pointer(&sysName[0])), 0)
	if err == _EINVAL {
		prSetVMAUnsupported.Store(true)
	}
	// ignore other errors
}

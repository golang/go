// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd || dragonfly
// +build freebsd dragonfly

package os

import (
	"syscall"
	"unsafe"
)

func executable() (string, error) {
	mib := [4]int32{_CTL_KERN, _KERN_PROC, _KERN_PROC_PATHNAME, -1}

	n := uintptr(0)
	// get length
	_, _, err := syscall.Syscall6(syscall.SYS___SYSCTL, uintptr(unsafe.Pointer(&mib[0])), 4, 0, uintptr(unsafe.Pointer(&n)), 0, 0)
	if err != 0 {
		return "", err
	}
	if n == 0 { // shouldn't happen
		return "", nil
	}
	buf := make([]byte, n)
	_, _, err = syscall.Syscall6(syscall.SYS___SYSCTL, uintptr(unsafe.Pointer(&mib[0])), 4, uintptr(unsafe.Pointer(&buf[0])), uintptr(unsafe.Pointer(&n)), 0, 0)
	if err != 0 {
		return "", err
	}
	if n == 0 { // shouldn't happen
		return "", nil
	}
	return string(buf[:n-1]), nil
}

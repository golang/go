// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"syscall"
	"unsafe"
)

const (
	_CTL_KERN = 1

	_KERN_ARND = 81
)

func Arandom(p []byte) error {
	mib := [2]uint32{_CTL_KERN, _KERN_ARND}
	n := uintptr(len(p))
	_, _, errno := syscall.Syscall6(
		syscall.SYS___SYSCTL,
		uintptr(unsafe.Pointer(&mib[0])),
		uintptr(len(mib)),
		uintptr(unsafe.Pointer(&p[0])), // olddata
		uintptr(unsafe.Pointer(&n)),    // &oldlen
		uintptr(unsafe.Pointer(nil)),   // newdata
		0)                              // newlen
	if errno != 0 {
		return syscall.Errno(errno)
	}
	if n != uintptr(len(p)) {
		return syscall.EINVAL
	}
	return nil
}

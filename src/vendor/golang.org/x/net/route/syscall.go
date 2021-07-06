// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || netbsd || openbsd
// +build dragonfly freebsd netbsd openbsd

package route

import (
	"syscall"
	"unsafe"
)

var zero uintptr

func sysctl(mib []int32, old *byte, oldlen *uintptr, new *byte, newlen uintptr) error {
	var p unsafe.Pointer
	if len(mib) > 0 {
		p = unsafe.Pointer(&mib[0])
	} else {
		p = unsafe.Pointer(&zero)
	}
	_, _, errno := syscall.Syscall6(syscall.SYS___SYSCTL, uintptr(p), uintptr(len(mib)), uintptr(unsafe.Pointer(old)), uintptr(unsafe.Pointer(oldlen)), uintptr(unsafe.Pointer(new)), newlen)
	if errno != 0 {
		return error(errno)
	}
	return nil
}

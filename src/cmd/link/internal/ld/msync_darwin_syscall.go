// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && !go1.20

package ld

import (
	"syscall"
	"unsafe"
)

func msync(b []byte, flags int) (err error) {
	var p unsafe.Pointer
	if len(b) > 0 {
		p = unsafe.Pointer(&b[0])
	}
	_, _, errno := syscall.Syscall(syscall.SYS_MSYNC, uintptr(p), uintptr(len(b)), uintptr(flags))
	if errno != 0 {
		return errno
	}
	return nil
}

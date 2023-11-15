// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux && gccgo && arm

package unix

import (
	"syscall"
	"unsafe"
)

func seek(fd int, offset int64, whence int) (int64, syscall.Errno) {
	var newoffset int64
	offsetLow := uint32(offset & 0xffffffff)
	offsetHigh := uint32((offset >> 32) & 0xffffffff)
	_, _, err := Syscall6(SYS__LLSEEK, uintptr(fd), uintptr(offsetHigh), uintptr(offsetLow), uintptr(unsafe.Pointer(&newoffset)), uintptr(whence), 0)
	return newoffset, err
}

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && go1.12 && !go1.13
// +build darwin,go1.12,!go1.13

package unix

import (
	"unsafe"
)

const _SYS_GETDIRENTRIES64 = 344

func Getdirentries(fd int, buf []byte, basep *uintptr) (n int, err error) {
	// To implement this using libSystem we'd need syscall_syscallPtr for
	// fdopendir. However, syscallPtr was only added in Go 1.13, so we fall
	// back to raw syscalls for this func on Go 1.12.
	var p unsafe.Pointer
	if len(buf) > 0 {
		p = unsafe.Pointer(&buf[0])
	} else {
		p = unsafe.Pointer(&_zero)
	}
	r0, _, e1 := Syscall6(_SYS_GETDIRENTRIES64, uintptr(fd), uintptr(p), uintptr(len(buf)), uintptr(unsafe.Pointer(basep)), 0, 0)
	n = int(r0)
	if e1 != 0 {
		return n, errnoErr(e1)
	}
	return n, nil
}

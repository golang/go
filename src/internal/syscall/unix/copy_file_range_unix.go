// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd || linux

package unix

import (
	"syscall"
	"unsafe"
)

func CopyFileRange(rfd int, roff *int64, wfd int, woff *int64, len int, flags int) (n int, err error) {
	r1, _, errno := syscall.Syscall6(copyFileRangeTrap,
		uintptr(rfd),
		uintptr(unsafe.Pointer(roff)),
		uintptr(wfd),
		uintptr(unsafe.Pointer(woff)),
		uintptr(len),
		uintptr(flags),
	)
	n = int(r1)
	if errno != 0 {
		err = errno
	}
	return
}

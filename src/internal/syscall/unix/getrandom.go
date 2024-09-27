// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || freebsd || linux

package unix

import (
	"sync/atomic"
	"syscall"
	"unsafe"
)

var getrandomUnsupported atomic.Bool

// GetRandomFlag is a flag supported by the getrandom system call.
type GetRandomFlag uintptr

// GetRandom calls the getrandom system call.
func GetRandom(p []byte, flags GetRandomFlag) (n int, err error) {
	if getrandomUnsupported.Load() {
		return 0, syscall.ENOSYS
	}
	r1, _, errno := syscall.Syscall(getrandomTrap,
		uintptr(unsafe.Pointer(unsafe.SliceData(p))),
		uintptr(len(p)),
		uintptr(flags))
	if errno != 0 {
		if errno == syscall.ENOSYS {
			getrandomUnsupported.Store(true)
		}
		return 0, errno
	}
	return int(r1), nil
}

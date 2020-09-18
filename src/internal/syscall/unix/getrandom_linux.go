// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"sync/atomic"
	"syscall"
	"unsafe"
)

var randomUnsupported int32 // atomic

// GetRandomFlag is a flag supported by the getrandom system call.
type GetRandomFlag uintptr

const (
	// GRND_NONBLOCK means return EAGAIN rather than blocking.
	GRND_NONBLOCK GetRandomFlag = 0x0001

	// GRND_RANDOM means use the /dev/random pool instead of /dev/urandom.
	GRND_RANDOM GetRandomFlag = 0x0002
)

// GetRandom calls the Linux getrandom system call.
// See https://git.kernel.org/cgit/linux/kernel/git/torvalds/linux.git/commit/?id=c6e9d6f38894798696f23c8084ca7edbf16ee895
func GetRandom(p []byte, flags GetRandomFlag) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if atomic.LoadInt32(&randomUnsupported) != 0 {
		return 0, syscall.ENOSYS
	}
	r1, _, errno := syscall.Syscall(getrandomTrap,
		uintptr(unsafe.Pointer(&p[0])),
		uintptr(len(p)),
		uintptr(flags))
	if errno != 0 {
		if errno == syscall.ENOSYS {
			atomic.StoreInt32(&randomUnsupported, 1)
		}
		return 0, errno
	}
	return int(r1), nil
}

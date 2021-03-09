// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"sync/atomic"
	"syscall"
	"unsafe"
)

var randomUnsupported int32 // atomic

// DragonFlyBSD getrandom system call number.
const randomTrap uintptr = 550

// GetRandomFlag is a flag supported by the getrandom system call.
type GetRandomFlag uintptr

const (
	// GRND_RANDOM is only set for portability purpose, no-op on DragonFlyBSD.
	GRND_RANDOM GetRandomFlag = 0x0001

	// GRND_NONBLOCK means return EAGAIN rather than blocking.
	GRND_NONBLOCK GetRandomFlag = 0x0002

	// GRND_INSECURE is an GRND_NONBLOCK alias
	GRND_INSECURE GetRandomFlag = 0x0004
)

// GetRandom calls the DragonFlyBSD getrandom system call.
func GetRandom(p []byte, flags GetRandomFlag) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if atomic.LoadInt32(&randomUnsupported) != 0 {
		return 0, syscall.ENOSYS
	}
	r1, _, errno := syscall.Syscall(randomTrap,
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

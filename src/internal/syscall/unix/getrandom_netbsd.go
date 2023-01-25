// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"
)

// NetBSD getrandom system call number.
const getrandomTrap uintptr = 91

var getrandomUnsupported int32 // atomic

// GetRandomFlag is a flag supported by the getrandom system call.
type GetRandomFlag uintptr

// GetRandom calls the getrandom system call.
func GetRandom(p []byte, flags GetRandomFlag) (n int, err error) {
	if len(p) == 0 {
		return 0, nil
	}
	if atomic.LoadInt32(&getrandomUnsupported) != 0 {
		return 0, syscall.ENOSYS
	}
	// getrandom(2) was added in NetBSD 10.0
	if getOSRevision() < 1000000000 {
		atomic.StoreInt32(&getrandomUnsupported, 1)
		return 0, syscall.ENOSYS
	}
	r1, _, errno := syscall.Syscall(getrandomTrap,
		uintptr(unsafe.Pointer(&p[0])),
		uintptr(len(p)),
		uintptr(flags))
	if errno != 0 {
		if errno == syscall.ENOSYS {
			atomic.StoreInt32(&getrandomUnsupported, 1)
		}
		return 0, errno
	}
	return int(r1), nil
}

var (
	osrevisionOnce sync.Once
	osrevision     uint32
)

func getOSRevision() uint32 {
	osrevisionOnce.Do(func() { osrevision, _ = syscall.SysctlUint32("kern.osrevision") })
	return osrevision
}

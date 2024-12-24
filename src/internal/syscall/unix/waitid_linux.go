// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package unix

import (
	"syscall"
	"unsafe"
)

const (
	P_PID   = 1
	P_PIDFD = 3
)

func Waitid(idType int, id int, info *SiginfoChild, options int, rusage *syscall.Rusage) error {
	_, _, errno := syscall.Syscall6(syscall.SYS_WAITID, uintptr(idType), uintptr(id), uintptr(unsafe.Pointer(info)), uintptr(options), uintptr(unsafe.Pointer(rusage)), 0)
	if errno != 0 {
		return errno
	}
	return nil
}

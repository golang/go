// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"syscall"
	"unsafe"
)

func (out *OutBuf) fallocate(size uint64) error {
	store := &syscall.Fstore_t{
		Flags:   syscall.F_ALLOCATEALL,
		Posmode: syscall.F_PEOFPOSMODE,
		Offset:  0,
		Length:  int64(size),
	}

	_, _, err := syscall.Syscall(syscall.SYS_FCNTL, uintptr(out.f.Fd()), syscall.F_PREALLOCATE, uintptr(unsafe.Pointer(store)))
	if err != 0 {
		return err
	}

	return nil
}

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"syscall"
	"unsafe"
)

func (out *OutBuf) fallocate(size uint64) error {
	stat, err := out.f.Stat()
	if err != nil {
		return err
	}
	cursize := uint64(stat.Size())
	if size <= cursize {
		return nil
	}

	store := &syscall.Fstore_t{
		Flags:   syscall.F_ALLOCATEALL,
		Posmode: syscall.F_PEOFPOSMODE,
		Offset:  0,
		Length:  int64(size - cursize), // F_PEOFPOSMODE allocates from the end of the file, so we want the size difference here
	}

	_, _, errno := syscall.Syscall(syscall.SYS_FCNTL, uintptr(out.f.Fd()), syscall.F_PREALLOCATE, uintptr(unsafe.Pointer(store)))
	if errno != 0 {
		return errno
	}

	return nil
}

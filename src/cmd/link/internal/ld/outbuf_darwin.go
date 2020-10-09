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
	// F_PEOFPOSMODE allocates from the end of the file, so we want the size difference.
	// Apparently, it uses the end of the allocation, instead of the logical end of the
	// the file.
	cursize := uint64(stat.Sys().(*syscall.Stat_t).Blocks * 512) // allocated size
	if size <= cursize {
		return nil
	}

	store := &syscall.Fstore_t{
		Flags:   syscall.F_ALLOCATEALL,
		Posmode: syscall.F_PEOFPOSMODE,
		Offset:  0,
		Length:  int64(size - cursize),
	}

	_, _, errno := syscall.Syscall(syscall.SYS_FCNTL, uintptr(out.f.Fd()), syscall.F_PREALLOCATE, uintptr(unsafe.Pointer(store)))
	if errno != 0 {
		return errno
	}

	return nil
}

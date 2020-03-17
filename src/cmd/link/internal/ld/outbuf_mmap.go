// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux openbsd

package ld

import (
	"syscall"
	"unsafe"
)

func (out *OutBuf) Mmap(filesize uint64) error {
	err := out.f.Truncate(int64(filesize))
	if err != nil {
		Exitf("resize output file failed: %v", err)
	}
	out.length = filesize
	out.buf, err = syscall.Mmap(int(out.f.Fd()), 0, int(filesize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED|syscall.MAP_FILE)
	return err
}

func (out *OutBuf) Munmap() {
	if out.buf == nil {
		return
	}
	err := out.Msync()
	if err != nil {
		Exitf("msync output file failed: %v", err)
	}
	syscall.Munmap(out.buf)
	out.buf = nil
	_, err = out.f.Seek(out.off, 0)
	if err != nil {
		Exitf("seek output file failed: %v", err)
	}
}

func (out *OutBuf) Msync() error {
	if out.buf == nil || out.length <= 0 {
		return nil
	}
	// TODO: netbsd supports mmap and msync, but the syscall package doesn't define MSYNC.
	// It is excluded from the build tag for now.
	_, _, errno := syscall.Syscall(syscall.SYS_MSYNC, uintptr(unsafe.Pointer(&out.buf[0])), uintptr(out.length), syscall.MS_SYNC)
	if errno != 0 {
		return errno
	}
	return nil
}

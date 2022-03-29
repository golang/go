// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd
// +build aix darwin dragonfly freebsd linux netbsd openbsd

package ld

import (
	"syscall"
)

// Mmap maps the output file with the given size. It unmaps the old mapping
// if it is already mapped. It also flushes any in-heap data to the new
// mapping.
func (out *OutBuf) Mmap(filesize uint64) (err error) {
	oldlen := len(out.buf)
	if oldlen != 0 {
		out.munmap()
	}

	for {
		if err = out.fallocate(filesize); err != syscall.EINTR {
			break
		}
	}
	if err != nil {
		// Some file systems do not support fallocate. We ignore that error as linking
		// can still take place, but you might SIGBUS when you write to the mmapped
		// area.
		if err != syscall.ENOTSUP && err != syscall.EPERM && err != errNoFallocate {
			return err
		}
	}
	err = out.f.Truncate(int64(filesize))
	if err != nil {
		Exitf("resize output file failed: %v", err)
	}
	out.buf, err = syscall.Mmap(int(out.f.Fd()), 0, int(filesize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED|syscall.MAP_FILE)
	if err != nil {
		return err
	}

	// copy heap to new mapping
	if uint64(oldlen+len(out.heap)) > filesize {
		panic("mmap size too small")
	}
	copy(out.buf[oldlen:], out.heap)
	out.heap = out.heap[:0]
	return nil
}

func (out *OutBuf) munmap() {
	if out.buf == nil {
		return
	}
	syscall.Munmap(out.buf)
	out.buf = nil
}

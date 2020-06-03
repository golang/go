// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux openbsd

package ld

import (
	"syscall"
)

func (out *OutBuf) Mmap(filesize uint64) (err error) {
	for {
		if err = out.fallocate(filesize); err != syscall.EINTR {
			break
		}
	}
	if err != nil {
		// Some file systems do not support fallocate. We ignore that error as linking
		// can still take place, but you might SIGBUS when you write to the mmapped
		// area.
		if err.Error() != fallocateNotSupportedErr {
			return err
		}
	}
	err = out.f.Truncate(int64(filesize))
	if err != nil {
		Exitf("resize output file failed: %v", err)
	}
	out.buf, err = syscall.Mmap(int(out.f.Fd()), 0, int(filesize), syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED|syscall.MAP_FILE)
	return err
}

func (out *OutBuf) munmap() {
	if out.buf == nil {
		return
	}
	syscall.Munmap(out.buf)
	out.buf = nil
	_, err := out.f.Seek(out.off, 0)
	if err != nil {
		Exitf("seek output file failed: %v", err)
	}
}

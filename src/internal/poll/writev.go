// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package poll

import (
	"io"
	"runtime"
	"syscall"
)

// Writev wraps the writev system call.
func (fd *FD) Writev(v *[][]byte) (int64, error) {
	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()
	if err := fd.pd.prepareWrite(fd.isFile); err != nil {
		return 0, err
	}

	var iovecs []syscall.Iovec
	if fd.iovecs != nil {
		iovecs = *fd.iovecs
	}
	// TODO: read from sysconf(_SC_IOV_MAX)? The Linux default is
	// 1024 and this seems conservative enough for now. Darwin's
	// UIO_MAXIOV also seems to be 1024.
	maxVec := 1024
	if runtime.GOOS == "aix" || runtime.GOOS == "solaris" {
		// IOV_MAX is set to XOPEN_IOV_MAX on AIX and Solaris.
		maxVec = 16
	}

	var n int64
	var err error
	for len(*v) > 0 {
		iovecs = iovecs[:0]
		for _, chunk := range *v {
			if len(chunk) == 0 {
				continue
			}
			iovecs = append(iovecs, newIovecWithBase(&chunk[0]))
			if fd.IsStream && len(chunk) > 1<<30 {
				iovecs[len(iovecs)-1].SetLen(1 << 30)
				break // continue chunk on next writev
			}
			iovecs[len(iovecs)-1].SetLen(len(chunk))
			if len(iovecs) == maxVec {
				break
			}
		}
		if len(iovecs) == 0 {
			break
		}
		if fd.iovecs == nil {
			fd.iovecs = new([]syscall.Iovec)
		}
		*fd.iovecs = iovecs // cache

		var wrote uintptr
		wrote, err = writev(fd.Sysfd, iovecs)
		if wrote == ^uintptr(0) {
			wrote = 0
		}
		TestHookDidWritev(int(wrote))
		n += int64(wrote)
		consume(v, int64(wrote))
		for i := range iovecs {
			iovecs[i] = syscall.Iovec{}
		}
		if err != nil {
			if err == syscall.EINTR {
				continue
			}
			if err == syscall.EAGAIN {
				if err = fd.pd.waitWrite(fd.isFile); err == nil {
					continue
				}
			}
			break
		}
		if n == 0 {
			err = io.ErrUnexpectedEOF
			break
		}
	}
	return n, err
}

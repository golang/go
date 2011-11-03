// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux openbsd

package os

import (
	"io"
	"syscall"
)

const (
	blockSize = 4096
)

// Readdirnames reads and returns a slice of names from the directory f.
//
// If n > 0, Readdirnames returns at most n names. In this case, if
// Readdirnames returns an empty slice, it will return a non-nil error
// explaining why. At the end of a directory, the error is io.EOF.
//
// If n <= 0, Readdirnames returns all the names from the directory in
// a single slice. In this case, if Readdirnames succeeds (reads all
// the way to the end of the directory), it returns the slice and a
// nil error. If it encounters an error before the end of the
// directory, Readdirnames returns the names read until that point and
// a non-nil error.
func (f *File) Readdirnames(n int) (names []string, err error) {
	// If this file has no dirinfo, create one.
	if f.dirinfo == nil {
		f.dirinfo = new(dirInfo)
		// The buffer must be at least a block long.
		f.dirinfo.buf = make([]byte, blockSize)
	}
	d := f.dirinfo

	size := n
	if size <= 0 {
		size = 100
		n = -1
	}

	names = make([]string, 0, size) // Empty with room to grow.
	for n != 0 {
		// Refill the buffer if necessary
		if d.bufp >= d.nbuf {
			d.bufp = 0
			var errno int
			d.nbuf, errno = syscall.ReadDirent(f.fd, d.buf)
			if errno != 0 {
				return names, NewSyscallError("readdirent", errno)
			}
			if d.nbuf <= 0 {
				break // EOF
			}
		}

		// Drain the buffer
		var nb, nc int
		nb, nc, names = syscall.ParseDirent(d.buf[d.bufp:d.nbuf], n, names)
		d.bufp += nb
		n -= nc
	}
	if n >= 0 && len(names) == 0 {
		return names, io.EOF
	}
	return names, nil
}

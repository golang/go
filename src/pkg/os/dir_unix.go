// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
)

const (
	blockSize = 4096
)

// Readdirnames reads the contents of the directory associated with file and
// returns an array of up to count names, in directory order.  Subsequent
// calls on the same file will yield further names.
// A negative count means to read until EOF.
// Readdirnames returns the array and an Error, if any.
func (file *File) Readdirnames(count int) (names []string, err Error) {
	// If this file has no dirinfo, create one.
	if file.dirinfo == nil {
		file.dirinfo = new(dirInfo)
		// The buffer must be at least a block long.
		file.dirinfo.buf = make([]byte, blockSize)
	}
	d := file.dirinfo
	size := count
	if size < 0 {
		size = 100
	}
	names = make([]string, 0, size) // Empty with room to grow.
	for count != 0 {
		// Refill the buffer if necessary
		if d.bufp >= d.nbuf {
			d.bufp = 0
			var errno int
			d.nbuf, errno = syscall.ReadDirent(file.fd, d.buf)
			if errno != 0 {
				return names, NewSyscallError("readdirent", errno)
			}
			if d.nbuf <= 0 {
				break // EOF
			}
		}

		// Drain the buffer
		var nb, nc int
		nb, nc, names = syscall.ParseDirent(d.buf[d.bufp:d.nbuf], count, names)
		d.bufp += nb
		count -= nc
	}
	return names, nil
}

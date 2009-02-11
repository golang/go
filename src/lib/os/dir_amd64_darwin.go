// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall";
	"os";
	"unsafe";
)

const (
	blockSize = 4096	// TODO(r): use statfs
)

// Negative count means read until EOF.
func Readdirnames(fd *FD, count int) (names []string, err *os.Error) {
	// If this fd has no dirinfo, create one.
	if fd.dirinfo == nil {
		fd.dirinfo = new(dirInfo);
		// The buffer must be at least a block long.
		// TODO(r): use fstatfs to find fs block size.
		fd.dirinfo.buf = make([]byte, blockSize);
	}
	d := fd.dirinfo;
	size := count;
	if size < 0 {
		size = 100
	}
	names = make([]string, 0, size);	// Empty with room to grow.
	for count != 0 {
		// Refill the buffer if necessary
		if d.bufp == d.nbuf {
			var errno int64;
			// Final argument is (basep *int64) and the syscall doesn't take nil.
			d.nbuf, errno = syscall.Getdirentries(fd.fd, &d.buf[0], int64(len(d.buf)), new(int64));
			if d.nbuf < 0 {
				return names, os.ErrnoToError(errno)
			}
			if d.nbuf == 0 {
				break	// EOF
			}
			d.bufp = 0;
		}
		// Drain the buffer
		for count != 0 && d.bufp < d.nbuf {
			dirent := unsafe.Pointer(&d.buf[d.bufp]).(*syscall.Dirent);
			d.bufp += int64(dirent.Reclen);
			if dirent.Ino == 0 {	// File absent in directory.
				continue
			}
			count--;
			if len(names) == cap(names) {
				nnames := make([]string, len(names), 2*len(names));
				for i := 0; i < len(names); i++ {
					nnames[i] = names[i]
				}
				names = nnames;
			}
			names = names[0:len(names)+1];
			names[len(names)-1] = string(dirent.Name[0:dirent.Namlen]);
		}
	}
	return names, nil
}

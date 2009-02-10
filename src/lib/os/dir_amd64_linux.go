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

func clen(n []byte) int {
	for i := 0; i < len(n); i++ {
		if n[i] == 0 {
			return i
		}
	}
	return len(n)
}

// Negative count means read until EOF.
func Readdirnames(fd *FD, count int) (names []string, err *os.Error) {
	// If this fd has no dirinfo, create one.
	if fd.dirinfo == nil {
		fd.dirinfo = new(DirInfo);
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
			dbuf := unsafe.Pointer(&d.buf[0]).(*syscall.Dirent);
			d.nbuf, errno = syscall.Getdents(fd.fd, dbuf, int64(len(d.buf)));
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
			names[len(names)-1] = string(dirent.Name[0:clen(dirent.Name)]);
		}
	}
	return names, nil;
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"os";
	"syscall";
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
func readdirnames(file *File, count int) (names []string, err Error) {
	// If this file has no dirinfo, create one.
	if file.dirinfo == nil {
		file.dirinfo = new(dirInfo);
		// The buffer must be at least a block long.
		// TODO(r): use fstatfs to find fs block size.
		file.dirinfo.buf = make([]byte, blockSize);
	}
	d := file.dirinfo;
	size := count;
	if size < 0 {
		size = 100
	}
	names = make([]string, 0, size);	// Empty with room to grow.
	for count != 0 {
		// Refill the buffer if necessary
		if d.bufp >= d.nbuf {
			var errno int;
			d.nbuf, errno = syscall.Getdents(file.fd, d.buf);
			if errno != 0 {
				return names, NewSyscallError("getdents", errno)
			}
			if d.nbuf <= 0 {
				break	// EOF
			}
			d.bufp = 0;
		}
		// Drain the buffer
		for count != 0 && d.bufp < d.nbuf {
			dirent := (*syscall.Dirent)(unsafe.Pointer(&d.buf[d.bufp]));
			d.bufp += int(dirent.Reclen);
			if dirent.Ino == 0 {	// File absent in directory.
				continue
			}
			bytes := (*[len(dirent.Name)]byte)(unsafe.Pointer(&dirent.Name[0]));
			var name = string(bytes[0:clen(bytes)]);
			if name == "." || name == ".." {	// Useless names
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
			names[len(names)-1] = name;
		}
	}
	return names, nil;
}

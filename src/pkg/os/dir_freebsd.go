// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall";
	"unsafe";
)

const (
	blockSize = 4096;	// TODO(r): use statfs
)

func (file *File) Readdirnames(count int) (names []string, err Error) {
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
			d.bufp = 0;
			// Final argument is (basep *uintptr) and the syscall doesn't take nil.
			d.nbuf, errno = syscall.Getdirentries(file.fd, d.buf, new(uintptr));
			if errno != 0 {
				d.nbuf = 0;
				return names, NewSyscallError("getdirentries", errno);
			}
			if d.nbuf <= 0 {
				break	// EOF
			}
		}
		// Drain the buffer
		for count != 0 && d.bufp < d.nbuf {
			dirent := (*syscall.Dirent)(unsafe.Pointer(&d.buf[d.bufp]));
			if dirent.Reclen == 0 {
				d.bufp = d.nbuf;
				break;
			}
			d.bufp += int(dirent.Reclen);
			if dirent.Fileno == 0 {	// File absent in directory.
				continue
			}
			bytes := (*[len(dirent.Name)]byte)(unsafe.Pointer(&dirent.Name[0]));
			var name = string(bytes[0:dirent.Namlen]);
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
			names = names[0 : len(names)+1];
			names[len(names)-1] = name;
		}
	}
	return names, nil;
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall";
	"os";
	"unsafe";
)

// Negative count means read until EOF.
func Readdirnames(fd *FD, count int) (names []string, err *os.Error) {
	// Getdirentries needs the file offset - it's too hard for the kernel to remember
	// a number it already has written down.
	base, err1 := syscall.Seek(fd.fd, 0, 1);
	if err1 != 0 {
		return nil, os.ErrnoToError(err1)
	}
	// The buffer must be at least a block long.
	// TODO(r): use fstatfs to find fs block size.
	var buf = make([]byte, 8192);
	names = make([]string, 0, 100);	// TODO: could be smarter about size
	for {
		if count == 0 {
			break
		}
		ret, err2 := syscall.Getdirentries(fd.fd, &buf[0], int64(len(buf)), &base);
		if ret < 0 || err2 != 0 {
			return names, os.ErrnoToError(err2)
		}
		if ret == 0 {
			break
		}
		for w, i := uintptr(0),uintptr(0); i < uintptr(ret); i += w {
			if count == 0 {
				break
			}
			dirent := unsafe.Pointer((uintptr(unsafe.Pointer(&buf[0])) + i)).(*syscall.Dirent);
			w = uintptr(dirent.Reclen);
			if dirent.Ino == 0 {
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
	return names, nil;
}

// TODO(r): see comment in dir_amd64_linux.go

// Negative count means read until EOF.
func Readdir(fd *FD, count int) (dirs []Dir, err *os.Error) {
	dirname := fd.name;
	if dirname == "" {
		dirname = ".";
	}
	dirname += "/";
	// Getdirentries needs the file offset - it's too hard for the kernel to remember
	// a number it already has written down.
	base, err1 := syscall.Seek(fd.fd, 0, 1);
	if err1 != 0 {
		return nil, os.ErrnoToError(err1)
	}
	// The buffer must be at least a block long.
	// TODO(r): use fstatfs to find fs block size.
	var buf = make([]byte, 8192);
	dirs = make([]Dir, 0, 100);	// TODO: could be smarter about size
	for {
		if count == 0 {
			break
		}
		ret, err2 := syscall.Getdirentries(fd.fd, &buf[0], int64(len(buf)), &base);
		if ret < 0 || err2 != 0 {
			return dirs, os.ErrnoToError(err2)
		}
		if ret == 0 {
			break
		}
		for w, i := uintptr(0),uintptr(0); i < uintptr(ret); i += w {
			if count == 0 {
				break
			}
			dirent := unsafe.Pointer((uintptr(unsafe.Pointer(&buf[0])) + i)).(*syscall.Dirent);
			w = uintptr(dirent.Reclen);
			if dirent.Ino == 0 {
				continue
			}
			count--;
			if len(dirs) == cap(dirs) {
				ndirs := make([]Dir, len(dirs), 2*len(dirs));
				for i := 0; i < len(dirs); i++ {
					ndirs[i] = dirs[i]
				}
				dirs = ndirs;
			}
			dirs = dirs[0:len(dirs)+1];
			filename := string(dirent.Name[0:dirent.Namlen]);
			dirp, err := Stat(dirname + filename);
			if dir == nil || err != nil {
				dirs[len(dirs)-1].Name = filename;	// rest will be zeroed out
			} else {
				dirs[len(dirs)-1] = *dirp;
			}
		}
	}
	return dirs, nil;
}

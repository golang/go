// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall";
	"os";
	"unsafe";
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
	// The buffer should be at least a block long.
	// TODO(r): use fstatfs to find fs block size.
	var buf = make([]syscall.Dirent, 8192/unsafe.Sizeof(*new(syscall.Dirent)));
	names = make([]string, 0, 100);	// TODO: could be smarter about size
	for {
		if count == 0 {
			break
		}
		ret, err2 := syscall.Getdents(fd.fd, &buf[0], int64(len(buf) * unsafe.Sizeof(buf[0])));
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
			names[len(names)-1] = string(dirent.Name[0:clen(dirent.Name)]);
		}
	}
	return names, nil;
}

// TODO(r): Readdir duplicates a lot of Readdirnames. The other way would
// be to have Readdir (which could then be portable) call Readdirnames and
// then do the Stats.  The existing design was chosen to avoid allocating a
// throwaway names array, but the issue should be revisited once we have
// a better handle on what that overhead is with a strong garbage collector.
// Also, it's possible given the nature of the Unix kernel that interleaving
// reads of the directory with stats (as done here) would work better than
// one big read of the directory followed by a long run of Stat calls.

// Negative count means read until EOF.
func Readdir(fd *FD, count int) (dirs []Dir, err *os.Error) {
	dirname := fd.name;
	if dirname == "" {
		dirname = ".";
	}
	dirname += "/";
	// The buffer must be at least a block long.
	// TODO(r): use fstatfs to find fs block size.
	var buf = make([]syscall.Dirent, 8192/unsafe.Sizeof(*new(syscall.Dirent)));
	dirs = make([]Dir, 0, 100);	// TODO: could be smarter about size
	for {
		if count == 0 {
			break
		}
		ret, err2 := syscall.Getdents(fd.fd, &buf[0], int64(len(buf) * unsafe.Sizeof(buf[0])));
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
			filename := string(dirent.Name[0:clen(dirent.Name)]);
			dirp, err := Stat(dirname + filename);
			if dirp ==  nil || err != nil {
				dirs[len(dirs)-1].Name = filename;	// rest will be zeroed out
			} else {
				dirs[len(dirs)-1] = *dirp;
			}
		}
	}
	return dirs, nil;
}

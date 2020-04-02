// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"io"
	"runtime"
	"syscall"
	"unsafe"
)

// Auxiliary information if the File describes a directory
type dirInfo struct {
	dir uintptr // Pointer to DIR structure from dirent.h
}

func (d *dirInfo) close() {
	if d.dir == 0 {
		return
	}
	closedir(d.dir)
	d.dir = 0
}

func (f *File) readdirnames(n int) (names []string, err error) {
	if f.dirinfo == nil {
		dir, call, errno := f.pfd.OpenDir()
		if errno != nil {
			return nil, wrapSyscallError(call, errno)
		}
		f.dirinfo = &dirInfo{
			dir: dir,
		}
	}
	d := f.dirinfo

	size := n
	if size <= 0 {
		size = 100
		n = -1
	}

	names = make([]string, 0, size)
	var dirent syscall.Dirent
	var entptr *syscall.Dirent
	for len(names) < size || n == -1 {
		if res := readdir_r(d.dir, &dirent, &entptr); res != 0 {
			return names, wrapSyscallError("readdir", syscall.Errno(res))
		}
		if entptr == nil { // EOF
			break
		}
		if dirent.Ino == 0 {
			continue
		}
		name := (*[len(syscall.Dirent{}.Name)]byte)(unsafe.Pointer(&dirent.Name))[:]
		for i, c := range name {
			if c == 0 {
				name = name[:i]
				break
			}
		}
		// Check for useless names before allocating a string.
		if string(name) == "." || string(name) == ".." {
			continue
		}
		names = append(names, string(name))
		runtime.KeepAlive(f)
	}
	if n >= 0 && len(names) == 0 {
		return names, io.EOF
	}
	return names, nil
}

// Implemented in syscall/syscall_darwin.go.

//go:linkname closedir syscall.closedir
func closedir(dir uintptr) (err error)

//go:linkname readdir_r syscall.readdir_r
func readdir_r(dir uintptr, entry *syscall.Dirent, result **syscall.Dirent) (res int)

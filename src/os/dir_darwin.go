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

func (f *File) readdir(n int, mode readdirMode) (names []string, dirents []DirEntry, infos []FileInfo, err error) {
	if f.dirinfo == nil {
		dir, call, errno := f.pfd.OpenDir()
		if errno != nil {
			return nil, nil, nil, &PathError{Op: call, Path: f.name, Err: errno}
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

	var dirent syscall.Dirent
	var entptr *syscall.Dirent
	for len(names)+len(dirents)+len(infos) < size || n == -1 {
		if errno := readdir_r(d.dir, &dirent, &entptr); errno != 0 {
			if errno == syscall.EINTR {
				continue
			}
			return names, dirents, infos, &PathError{Op: "readdir", Path: f.name, Err: errno}
		}
		if entptr == nil { // EOF
			break
		}
		// Darwin may return a zero inode when a directory entry has been
		// deleted but not yet removed from the directory. The man page for
		// getdirentries(2) states that programs are responsible for skipping
		// those entries:
		//
		//   Users of getdirentries() should skip entries with d_fileno = 0,
		//   as such entries represent files which have been deleted but not
		//   yet removed from the directory entry.
		//
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
		if mode == readdirName {
			names = append(names, string(name))
		} else if mode == readdirDirEntry {
			de, err := newUnixDirent(f.name, string(name), dtToType(dirent.Type))
			if IsNotExist(err) {
				// File disappeared between readdir and stat.
				// Treat as if it didn't exist.
				continue
			}
			if err != nil {
				return nil, dirents, nil, err
			}
			dirents = append(dirents, de)
		} else {
			info, err := lstat(f.name + "/" + string(name))
			if IsNotExist(err) {
				// File disappeared between readdir + stat.
				// Treat as if it didn't exist.
				continue
			}
			if err != nil {
				return nil, nil, infos, err
			}
			infos = append(infos, info)
		}
		runtime.KeepAlive(f)
	}

	if n > 0 && len(names)+len(dirents)+len(infos) == 0 {
		return nil, nil, nil, io.EOF
	}
	return names, dirents, infos, nil
}

func dtToType(typ uint8) FileMode {
	switch typ {
	case syscall.DT_BLK:
		return ModeDevice
	case syscall.DT_CHR:
		return ModeDevice | ModeCharDevice
	case syscall.DT_DIR:
		return ModeDir
	case syscall.DT_FIFO:
		return ModeNamedPipe
	case syscall.DT_LNK:
		return ModeSymlink
	case syscall.DT_REG:
		return 0
	case syscall.DT_SOCK:
		return ModeSocket
	}
	return ^FileMode(0)
}

// Implemented in syscall/syscall_darwin.go.

//go:linkname closedir syscall.closedir
func closedir(dir uintptr) (err error)

//go:linkname readdir_r syscall.readdir_r
func readdir_r(dir uintptr, entry *syscall.Dirent, result **syscall.Dirent) (res syscall.Errno)

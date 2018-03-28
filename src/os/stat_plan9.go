// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
)

const bitSize16 = 2

func fileInfoFromStat(d *syscall.Dir) FileInfo {
	fs := &fileStat{
		name:    d.Name,
		size:    d.Length,
		modTime: time.Unix(int64(d.Mtime), 0),
		sys:     d,
	}
	fs.mode = FileMode(d.Mode & 0777)
	if d.Mode&syscall.DMDIR != 0 {
		fs.mode |= ModeDir
	}
	if d.Mode&syscall.DMAPPEND != 0 {
		fs.mode |= ModeAppend
	}
	if d.Mode&syscall.DMEXCL != 0 {
		fs.mode |= ModeExclusive
	}
	if d.Mode&syscall.DMTMP != 0 {
		fs.mode |= ModeTemporary
	}
	// Consider all files not served by #M as device files.
	if d.Type != 'M' {
		fs.mode |= ModeDevice
	}
	// Consider all files served by #c as character device files.
	if d.Type == 'c' {
		fs.mode |= ModeCharDevice
	}
	return fs
}

// arg is an open *File or a path string.
func dirstat(arg interface{}) (*syscall.Dir, error) {
	var name string
	var err error

	size := syscall.STATFIXLEN + 16*4

	for i := 0; i < 2; i++ {
		buf := make([]byte, bitSize16+size)

		var n int
		switch a := arg.(type) {
		case *File:
			name = a.name
			n, err = syscall.Fstat(a.fd, buf)
		case string:
			name = a
			n, err = syscall.Stat(a, buf)
		default:
			panic("phase error in dirstat")
		}

		if n < bitSize16 {
			return nil, &PathError{"stat", name, err}
		}

		// Pull the real size out of the stat message.
		size = int(uint16(buf[0]) | uint16(buf[1])<<8)

		// If the stat message is larger than our buffer we will
		// go around the loop and allocate one that is big enough.
		if size <= n {
			d, err := syscall.UnmarshalDir(buf[:n])
			if err != nil {
				return nil, &PathError{"stat", name, err}
			}
			return d, nil
		}

	}

	if err == nil {
		err = syscall.ErrBadStat
	}

	return nil, &PathError{"stat", name, err}
}

// statNolog implements Stat for Plan 9.
func statNolog(name string) (FileInfo, error) {
	d, err := dirstat(name)
	if err != nil {
		return nil, err
	}
	return fileInfoFromStat(d), nil
}

// lstatNolog implements Lstat for Plan 9.
func lstatNolog(name string) (FileInfo, error) {
	return statNolog(name)
}

// For testing.
func atime(fi FileInfo) time.Time {
	return time.Unix(int64(fi.Sys().(*syscall.Dir).Atime), 0)
}

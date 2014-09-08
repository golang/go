// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
)

func sameFile(fs1, fs2 *fileStat) bool {
	a := fs1.sys.(*syscall.Dir)
	b := fs2.sys.(*syscall.Dir)
	return a.Qid.Path == b.Qid.Path && a.Type == b.Type && a.Dev == b.Dev
}

func fileInfoFromStat(d *syscall.Dir) FileInfo {
	fs := &fileStat{
		name:    d.Name,
		size:    int64(d.Length),
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
	return fs
}

// arg is an open *File or a path string.
func dirstat(arg interface{}) (*syscall.Dir, error) {
	var name string

	// This is big enough for most stat messages
	// and rounded to a multiple of 128 bytes.
	size := (syscall.STATFIXLEN + 16*4 + 128) &^ 128

	for i := 0; i < 2; i++ {
		buf := make([]byte, size)

		var n int
		var err error
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
		if err != nil {
			return nil, &PathError{"stat", name, err}
		}
		if n < syscall.STATFIXLEN {
			return nil, &PathError{"stat", name, syscall.ErrShortStat}
		}

		// Pull the real size out of the stat message.
		size = int(uint16(buf[0]) | uint16(buf[1])<<8)

		// If the stat message is larger than our buffer we will
		// go around the loop and allocate one that is big enough.
		if size > n {
			continue
		}

		d, err := syscall.UnmarshalDir(buf[:n])
		if err != nil {
			return nil, &PathError{"stat", name, err}
		}
		return d, nil
	}
	return nil, &PathError{"stat", name, syscall.ErrBadStat}
}

// Stat returns a FileInfo describing the named file.
// If there is an error, it will be of type *PathError.
func Stat(name string) (fi FileInfo, err error) {
	d, err := dirstat(name)
	if err != nil {
		return nil, err
	}
	return fileInfoFromStat(d), nil
}

// Lstat returns a FileInfo describing the named file.
// If the file is a symbolic link, the returned FileInfo
// describes the symbolic link.  Lstat makes no attempt to follow the link.
// If there is an error, it will be of type *PathError.
func Lstat(name string) (fi FileInfo, err error) {
	return Stat(name)
}

// For testing.
func atime(fi FileInfo) time.Time {
	return time.Unix(int64(fi.Sys().(*syscall.Dir).Atime), 0)
}

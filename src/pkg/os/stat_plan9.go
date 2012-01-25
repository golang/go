// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
)

func sameFile(fs1, fs2 *FileStat) bool {
	a := fs1.Sys.(*Dir)
	b := fs2.Sys.(*Dir)
	return a.Qid.Path == b.Qid.Path && a.Type == b.Type && a.Dev == b.Dev
}

func fileInfoFromStat(d *Dir) FileInfo {
	fs := &FileStat{
		name:    d.Name,
		size:    int64(d.Length),
		modTime: time.Unix(int64(d.Mtime), 0),
		Sys:     d,
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
func dirstat(arg interface{}) (d *Dir, err error) {
	var name string

	// This is big enough for most stat messages
	// and rounded to a multiple of 128 bytes.
	size := (syscall.STATFIXLEN + 16*4 + 128) &^ 128

	for i := 0; i < 2; i++ {
		buf := make([]byte, size)

		var n int
		switch a := arg.(type) {
		case *File:
			name = a.name
			n, err = syscall.Fstat(a.fd, buf)
		case string:
			name = a
			n, err = syscall.Stat(name, buf)
		}
		if err != nil {
			return nil, &PathError{"stat", name, err}
		}
		if n < syscall.STATFIXLEN {
			return nil, &PathError{"stat", name, Eshortstat}
		}

		// Pull the real size out of the stat message.
		s, _ := gbit16(buf)
		size = int(s)

		// If the stat message is larger than our buffer we will
		// go around the loop and allocate one that is big enough.
		if size <= n {
			d, err = UnmarshalDir(buf[:n])
			if err != nil {
				return nil, &PathError{"stat", name, err}
			}
			return
		}
	}
	return nil, &PathError{"stat", name, Ebadstat}
}

// Stat returns a FileInfo structure describing the named file and an error, if any.
func Stat(name string) (FileInfo, error) {
	d, err := dirstat(name)
	if err != nil {
		return nil, err
	}
	return fileInfoFromStat(d), nil
}

// Lstat returns the FileInfo structure describing the named file and an
// error, if any.  If the file is a symbolic link (though Plan 9 does not have symbolic links), 
// the returned FileInfo describes the symbolic link.  Lstat makes no attempt to follow the link.
func Lstat(name string) (FileInfo, error) {
	return Stat(name)
}

// For testing.
func atime(fi FileInfo) time.Time {
	return time.Unix(int64(fi.(*FileStat).Sys.(*Dir).Atime), 0)
}

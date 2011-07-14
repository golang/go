// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import "syscall"

func fileInfoFromStat(fi *FileInfo, d *Dir) *FileInfo {
	fi.Dev = uint64(d.Qid.Vers) | uint64(d.Qid.Type<<32)
	fi.Ino = d.Qid.Path

	fi.Mode = uint32(d.Mode) & 0777
	if (d.Mode & syscall.DMDIR) == syscall.DMDIR {
		fi.Mode |= syscall.S_IFDIR
	} else {
		fi.Mode |= syscall.S_IFREG
	}

	fi.Size = int64(d.Length)
	fi.Atime_ns = 1e9 * int64(d.Atime)
	fi.Mtime_ns = 1e9 * int64(d.Mtime)
	fi.Name = d.Name
	fi.FollowedSymlink = false
	return fi
}

// arg is an open *File or a path string. 
func dirstat(arg interface{}) (d *Dir, err Error) {
	var name string
	nd := syscall.STATFIXLEN + 16*4

	for i := 0; i < 2; i++ { /* should work by the second try */
		buf := make([]byte, nd)

		var n int
		var e syscall.Error

		switch syscallArg := arg.(type) {
		case *File:
			name = syscallArg.name
			n, e = syscall.Fstat(syscallArg.fd, buf)
		case string:
			name = syscallArg
			n, e = syscall.Stat(name, buf)
		}

		if e != nil {
			return nil, &PathError{"stat", name, e}
		}

		if n < syscall.STATFIXLEN {
			return nil, &PathError{"stat", name, Eshortstat}
		}

		ntmp, _ := gbit16(buf)
		nd = int(ntmp)

		if nd <= n {
			d, e := UnmarshalDir(buf[:n])

			if e != nil {
				return nil, &PathError{"stat", name, e}
			}
			return d, e
		}
	}

	return nil, &PathError{"stat", name, Ebadstat}
}

// Stat returns a FileInfo structure describing the named file and an error, if any.
func Stat(name string) (fi *FileInfo, err Error) {
	d, err := dirstat(name)
	if iserror(err) {
		return nil, err
	}
	return fileInfoFromStat(new(FileInfo), d), err
}

// Lstat returns the FileInfo structure describing the named file and an
// error, if any.  If the file is a symbolic link (though Plan 9 does not have symbolic links), 
// the returned FileInfo describes the symbolic link.  Lstat makes no attempt to follow the link.
func Lstat(name string) (fi *FileInfo, err Error) {
	d, err := dirstat(name)
	if iserror(err) {
		return nil, err
	}
	return fileInfoFromStat(new(FileInfo), d), err
}

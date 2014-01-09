// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
)

func sameFile(fs1, fs2 *fileStat) bool {
	stat1 := fs1.sys.(*syscall.Stat_t)
	stat2 := fs2.sys.(*syscall.Stat_t)
	return stat1.Dev == stat2.Dev && stat1.Ino == stat2.Ino
}

func fileInfoFromStat(st *syscall.Stat_t, name string) FileInfo {
	fs := &fileStat{
		name:    basename(name),
		size:    int64(st.Size),
		modTime: timespecToTime(st.Mtim),
		sys:     st,
	}
	fs.mode = FileMode(st.Mode & 0777)
	switch st.Mode & syscall.S_IFMT {
	case syscall.S_IFBLK:
		fs.mode |= ModeDevice
	case syscall.S_IFCHR:
		fs.mode |= ModeDevice | ModeCharDevice
	case syscall.S_IFDIR:
		fs.mode |= ModeDir
	case syscall.S_IFIFO:
		fs.mode |= ModeNamedPipe
	case syscall.S_IFLNK:
		fs.mode |= ModeSymlink
	case syscall.S_IFREG:
		// nothing to do
	case syscall.S_IFSOCK:
		fs.mode |= ModeSocket
	}
	if st.Mode&syscall.S_ISGID != 0 {
		fs.mode |= ModeSetgid
	}
	if st.Mode&syscall.S_ISUID != 0 {
		fs.mode |= ModeSetuid
	}
	if st.Mode&syscall.S_ISVTX != 0 {
		fs.mode |= ModeSticky
	}
	return fs
}

func timespecToTime(ts syscall.Timespec) time.Time {
	return time.Unix(int64(ts.Sec), int64(ts.Nsec))
}

// For testing.
func atime(fi FileInfo) time.Time {
	return timespecToTime(fi.Sys().(*syscall.Stat_t).Atim)
}

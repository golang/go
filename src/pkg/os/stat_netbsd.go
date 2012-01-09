// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"time"
)

func sameFile(fs1, fs2 *FileStat) bool {
	sys1 := fs1.Sys.(*syscall.Stat_t)
	sys2 := fs2.Sys.(*syscall.Stat_t)
	return sys1.Dev == sys2.Dev && sys1.Ino == sys2.Ino
}

func fileInfoFromStat(st *syscall.Stat_t, name string) FileInfo {
	fs := &FileStat{
		name:    basename(name),
		size:    int64(st.Size),
		modTime: timespecToTime(st.Mtim),
		Sys:     st,
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
	return fs
}

func timespecToTime(ts syscall.Timespec) time.Time {
	return time.Unix(int64(ts.Sec), int64(ts.Nsec))
}

// For testing.
func atime(fi FileInfo) time.Time {
	return timespecToTime(fi.(*FileStat).Sys.(*syscall.Stat_t).Atim)
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build js,wasm nacl

package os

import (
	"syscall"
	"time"
)

func fillFileStatFromSys(fs *fileStat, name string) {
	fs.name = basename(name)
	fs.size = fs.sys.Size
	fs.modTime = timespecToTime(fs.sys.Mtime, fs.sys.MtimeNsec)
	fs.mode = FileMode(fs.sys.Mode & 0777)
	switch fs.sys.Mode & syscall.S_IFMT {
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
	if fs.sys.Mode&syscall.S_ISGID != 0 {
		fs.mode |= ModeSetgid
	}
	if fs.sys.Mode&syscall.S_ISUID != 0 {
		fs.mode |= ModeSetuid
	}
	if fs.sys.Mode&syscall.S_ISVTX != 0 {
		fs.mode |= ModeSticky
	}
}

func timespecToTime(sec, nsec int64) time.Time {
	return time.Unix(sec, nsec)
}

// For testing.
func atime(fi FileInfo) time.Time {
	st := fi.Sys().(*syscall.Stat_t)
	return timespecToTime(st.Atime, st.AtimeNsec)
}

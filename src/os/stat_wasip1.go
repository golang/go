// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package os

import (
	"syscall"
	"time"
)

func fillFileStatFromSys(fs *fileStat, name string) {
	fs.name = basename(name)
	fs.size = int64(fs.sys.Size)
	fs.mode = FileMode(fs.sys.Mode)
	fs.modTime = time.Unix(0, int64(fs.sys.Mtime))

	switch fs.sys.Filetype {
	case syscall.FILETYPE_BLOCK_DEVICE:
		fs.mode |= ModeDevice
	case syscall.FILETYPE_CHARACTER_DEVICE:
		fs.mode |= ModeDevice | ModeCharDevice
	case syscall.FILETYPE_DIRECTORY:
		fs.mode |= ModeDir
	case syscall.FILETYPE_SOCKET_DGRAM:
		fs.mode |= ModeSocket
	case syscall.FILETYPE_SOCKET_STREAM:
		fs.mode |= ModeSocket
	case syscall.FILETYPE_SYMBOLIC_LINK:
		fs.mode |= ModeSymlink
	}
}

// For testing.
func atime(fi FileInfo) time.Time {
	st := fi.Sys().(*syscall.Stat_t)
	return time.Unix(0, int64(st.Atime))
}

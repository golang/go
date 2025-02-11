// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package unix

import (
	"syscall"
	"unsafe"
)

//go:wasmimport wasi_snapshot_preview1 path_filestat_set_times
//go:noescape
func path_filestat_set_times(fd int32, flags uint32, path *byte, pathLen size, atim uint64, mtim uint64, fstflags uint32) syscall.Errno

func Utimensat(dirfd int, path string, times *[2]syscall.Timespec, flag int) error {
	if path == "" {
		return syscall.EINVAL
	}
	atime := syscall.TimespecToNsec(times[0])
	mtime := syscall.TimespecToNsec(times[1])

	var fflag uint32
	if times[0].Nsec != UTIME_OMIT {
		fflag |= syscall.FILESTAT_SET_ATIM
	}
	if times[1].Nsec != UTIME_OMIT {
		fflag |= syscall.FILESTAT_SET_MTIM
	}
	errno := path_filestat_set_times(
		int32(dirfd),
		syscall.LOOKUP_SYMLINK_FOLLOW,
		unsafe.StringData(path),
		size(len(path)),
		uint64(atime),
		uint64(mtime),
		fflag,
	)
	return errnoErr(errno)
}

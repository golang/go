// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package unix

import (
	"syscall"
	"unsafe"
)

const (
	// UTIME_OMIT is the sentinel value to indicate that a time value should not
	// be changed. It is useful for example to indicate for example with UtimesNano
	// to avoid changing AccessTime or ModifiedTime.
	// Its value must match syscall/fs_wasip1.go
	UTIME_OMIT = -0x2

	AT_REMOVEDIR = 0x200
)

func Unlinkat(dirfd int, path string, flags int) error {
	if flags&AT_REMOVEDIR == 0 {
		return errnoErr(path_unlink_file(
			int32(dirfd),
			unsafe.StringData(path),
			size(len(path)),
		))
	} else {
		return errnoErr(path_remove_directory(
			int32(dirfd),
			unsafe.StringData(path),
			size(len(path)),
		))
	}
}

//go:wasmimport wasi_snapshot_preview1 path_unlink_file
//go:noescape
func path_unlink_file(fd int32, path *byte, pathLen size) syscall.Errno

//go:wasmimport wasi_snapshot_preview1 path_remove_directory
//go:noescape
func path_remove_directory(fd int32, path *byte, pathLen size) syscall.Errno

func Openat(dirfd int, path string, flags int, perm uint32) (int, error) {
	return syscall.Openat(dirfd, path, flags, perm)
}

func Readlinkat(dirfd int, path string, buf []byte) (int, error) {
	var nwritten size
	errno := path_readlink(
		int32(dirfd),
		unsafe.StringData(path),
		size(len(path)),
		&buf[0],
		size(len(buf)),
		&nwritten)
	return int(nwritten), errnoErr(errno)

}

type (
	size = uint32
)

//go:wasmimport wasi_snapshot_preview1 path_readlink
//go:noescape
func path_readlink(fd int32, path *byte, pathLen size, buf *byte, bufLen size, nwritten *size) syscall.Errno

func Mkdirat(dirfd int, path string, mode uint32) error {
	if path == "" {
		return syscall.EINVAL
	}
	return errnoErr(path_create_directory(
		int32(dirfd),
		unsafe.StringData(path),
		size(len(path)),
	))
}

//go:wasmimport wasi_snapshot_preview1 path_create_directory
//go:noescape
func path_create_directory(fd int32, path *byte, pathLen size) syscall.Errno

func errnoErr(errno syscall.Errno) error {
	if errno == 0 {
		return nil
	}
	return errno
}

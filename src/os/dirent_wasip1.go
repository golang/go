// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package os

import (
	"internal/byteorder"
	"syscall"
	"unsafe"
)

// https://github.com/WebAssembly/WASI/blob/main/legacy/preview1/docs.md#-dirent-record
const sizeOfDirent = 24

func direntIno(buf []byte) (uint64, bool) {
	return byteorder.ReadInt(buf, unsafe.Offsetof(syscall.Dirent{}.Ino), unsafe.Sizeof(syscall.Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	namelen, ok := direntNamlen(buf)
	return sizeOfDirent + namelen, ok
}

func direntNamlen(buf []byte) (uint64, bool) {
	return byteorder.ReadInt(buf, unsafe.Offsetof(syscall.Dirent{}.Namlen), unsafe.Sizeof(syscall.Dirent{}.Namlen))
}

func direntType(buf []byte) FileMode {
	off := unsafe.Offsetof(syscall.Dirent{}.Type)
	if off >= uintptr(len(buf)) {
		return ^FileMode(0) // unknown
	}
	switch syscall.Filetype(buf[off]) {
	case syscall.FILETYPE_BLOCK_DEVICE:
		return ModeDevice
	case syscall.FILETYPE_CHARACTER_DEVICE:
		return ModeDevice | ModeCharDevice
	case syscall.FILETYPE_DIRECTORY:
		return ModeDir
	case syscall.FILETYPE_REGULAR_FILE:
		return 0
	case syscall.FILETYPE_SOCKET_DGRAM:
		return ModeSocket
	case syscall.FILETYPE_SOCKET_STREAM:
		return ModeSocket
	case syscall.FILETYPE_SYMBOLIC_LINK:
		return ModeSymlink
	}
	return ^FileMode(0) // unknown
}

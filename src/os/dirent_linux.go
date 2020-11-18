// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"syscall"
	"unsafe"
)

func direntIno(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(syscall.Dirent{}.Ino), unsafe.Sizeof(syscall.Dirent{}.Ino))
}

func direntReclen(buf []byte) (uint64, bool) {
	return readInt(buf, unsafe.Offsetof(syscall.Dirent{}.Reclen), unsafe.Sizeof(syscall.Dirent{}.Reclen))
}

func direntNamlen(buf []byte) (uint64, bool) {
	reclen, ok := direntReclen(buf)
	if !ok {
		return 0, false
	}
	return reclen - uint64(unsafe.Offsetof(syscall.Dirent{}.Name)), true
}

func direntType(buf []byte) FileMode {
	off := unsafe.Offsetof(syscall.Dirent{}.Type)
	if off >= uintptr(len(buf)) {
		return ^FileMode(0) // unknown
	}
	typ := buf[off]
	switch typ {
	case syscall.DT_BLK:
		return ModeDevice
	case syscall.DT_CHR:
		return ModeDevice | ModeCharDevice
	case syscall.DT_DIR:
		return ModeDir
	case syscall.DT_FIFO:
		return ModeNamedPipe
	case syscall.DT_LNK:
		return ModeSymlink
	case syscall.DT_REG:
		return 0
	case syscall.DT_SOCK:
		return ModeSocket
	}
	return ^FileMode(0) // unknown
}

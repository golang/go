// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux darwin freebsd openbsd netbsd
// +build !appengine

package imports

import (
	"bytes"
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

const blockSize = 8 << 10

// unknownFileMode is a sentinel (and bogus) os.FileMode
// value used to represent a syscall.DT_UNKNOWN Dirent.Type.
const unknownFileMode os.FileMode = os.ModeNamedPipe | os.ModeSocket | os.ModeDevice

func readDir(dirName string, fn func(dirName, entName string, typ os.FileMode) error) error {
	fd, err := syscall.Open(dirName, 0, 0)
	if err != nil {
		return &os.PathError{Op: "open", Path: dirName, Err: err}
	}
	defer syscall.Close(fd)

	// The buffer must be at least a block long.
	buf := make([]byte, blockSize) // stack-allocated; doesn't escape
	bufp := 0                      // starting read position in buf
	nbuf := 0                      // end valid data in buf
	for {
		if bufp >= nbuf {
			bufp = 0
			nbuf, err = syscall.ReadDirent(fd, buf)
			if err != nil {
				return os.NewSyscallError("readdirent", err)
			}
			if nbuf <= 0 {
				return nil
			}
		}
		consumed, name, typ := parseDirEnt(buf[bufp:nbuf])
		bufp += consumed
		if name == "" || name == "." || name == ".." {
			continue
		}
		// Fallback for filesystems (like old XFS) that don't
		// support Dirent.Type and have DT_UNKNOWN (0) there
		// instead.
		if typ == unknownFileMode {
			fi, err := os.Lstat(dirName + "/" + name)
			if err != nil {
				// It got deleted in the meantime.
				if os.IsNotExist(err) {
					continue
				}
				return err
			}
			typ = fi.Mode() & os.ModeType
		}
		if err := fn(dirName, name, typ); err != nil {
			return err
		}
	}
}

func parseDirEnt(buf []byte) (consumed int, name string, typ os.FileMode) {
	// golang.org/issue/15653
	dirent := (*syscall.Dirent)(unsafe.Pointer(&buf[0]))
	if v := unsafe.Offsetof(dirent.Reclen) + unsafe.Sizeof(dirent.Reclen); uintptr(len(buf)) < v {
		panic(fmt.Sprintf("buf size of %d smaller than dirent header size %d", len(buf), v))
	}
	if len(buf) < int(dirent.Reclen) {
		panic(fmt.Sprintf("buf size %d < record length %d", len(buf), dirent.Reclen))
	}
	consumed = int(dirent.Reclen)
	if direntInode(dirent) == 0 { // File absent in directory.
		return
	}
	switch dirent.Type {
	case syscall.DT_REG:
		typ = 0
	case syscall.DT_DIR:
		typ = os.ModeDir
	case syscall.DT_LNK:
		typ = os.ModeSymlink
	case syscall.DT_BLK:
		typ = os.ModeDevice
	case syscall.DT_FIFO:
		typ = os.ModeNamedPipe
	case syscall.DT_SOCK:
		typ = os.ModeSocket
	case syscall.DT_UNKNOWN:
		typ = unknownFileMode
	default:
		// Skip weird things.
		// It's probably a DT_WHT (http://lwn.net/Articles/325369/)
		// or something. Revisit if/when this package is moved outside
		// of goimports. goimports only cares about regular files,
		// symlinks, and directories.
		return
	}

	nameBuf := (*[unsafe.Sizeof(dirent.Name)]byte)(unsafe.Pointer(&dirent.Name[0]))
	nameLen := bytes.IndexByte(nameBuf[:], 0)
	if nameLen < 0 {
		panic("failed to find terminating 0 byte in dirent")
	}

	// Special cases for common things:
	if nameLen == 1 && nameBuf[0] == '.' {
		name = "."
	} else if nameLen == 2 && nameBuf[0] == '.' && nameBuf[1] == '.' {
		name = ".."
	} else {
		name = string(nameBuf[:nameLen])
	}
	return
}

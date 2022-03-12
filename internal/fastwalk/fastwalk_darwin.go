// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && cgo
// +build darwin,cgo

package fastwalk

/*
#include <dirent.h>

// fastwalk_readdir_r wraps readdir_r so that we don't have to pass a dirent**
// result pointer which triggers CGO's "Go pointer to Go pointer" check unless
// we allocat the result dirent* with malloc.
//
// fastwalk_readdir_r returns 0 on success, -1 upon reaching the end of the
// directory, or a positive error number to indicate failure.
static int fastwalk_readdir_r(DIR *fd, struct dirent *entry) {
	struct dirent *result;
	int ret = readdir_r(fd, entry, &result);
	if (ret == 0 && result == NULL) {
		ret = -1; // EOF
	}
	return ret;
}
*/
import "C"

import (
	"os"
	"syscall"
	"unsafe"
)

func readDir(dirName string, fn func(dirName, entName string, typ os.FileMode) error) error {
	fd, err := openDir(dirName)
	if err != nil {
		return &os.PathError{Op: "opendir", Path: dirName, Err: err}
	}
	defer C.closedir(fd)

	skipFiles := false
	var dirent syscall.Dirent
	for {
		ret := int(C.fastwalk_readdir_r(fd, (*C.struct_dirent)(unsafe.Pointer(&dirent))))
		if ret != 0 {
			if ret == -1 {
				break // EOF
			}
			if ret == int(syscall.EINTR) {
				continue
			}
			return &os.PathError{Op: "readdir", Path: dirName, Err: syscall.Errno(ret)}
		}
		if dirent.Ino == 0 {
			continue
		}
		typ := dtToType(dirent.Type)
		if skipFiles && typ.IsRegular() {
			continue
		}
		name := (*[len(syscall.Dirent{}.Name)]byte)(unsafe.Pointer(&dirent.Name))[:]
		name = name[:dirent.Namlen]
		for i, c := range name {
			if c == 0 {
				name = name[:i]
				break
			}
		}
		// Check for useless names before allocating a string.
		if string(name) == "." || string(name) == ".." {
			continue
		}
		if err := fn(dirName, string(name), typ); err != nil {
			if err != ErrSkipFiles {
				return err
			}
			skipFiles = true
		}
	}

	return nil
}

func dtToType(typ uint8) os.FileMode {
	switch typ {
	case syscall.DT_BLK:
		return os.ModeDevice
	case syscall.DT_CHR:
		return os.ModeDevice | os.ModeCharDevice
	case syscall.DT_DIR:
		return os.ModeDir
	case syscall.DT_FIFO:
		return os.ModeNamedPipe
	case syscall.DT_LNK:
		return os.ModeSymlink
	case syscall.DT_REG:
		return 0
	case syscall.DT_SOCK:
		return os.ModeSocket
	}
	return ^os.FileMode(0)
}

// openDir wraps opendir(3) and handles any EINTR errors. The returned *DIR
// needs to be closed with closedir(3).
func openDir(path string) (*C.DIR, error) {
	name, err := syscall.BytePtrFromString(path)
	if err != nil {
		return nil, err
	}
	for {
		fd, err := C.opendir((*C.char)(unsafe.Pointer(name)))
		if err != syscall.EINTR {
			return fd, err
		}
	}
}

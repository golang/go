// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package os

import (
	"internal/poll"
	"syscall"
)

func open(filePath string, flag int, perm uint32) (int, poll.SysFile, error) {
	if filePath == "" {
		return -1, poll.SysFile{}, syscall.EINVAL
	}
	absPath := filePath
	// os.(*File).Chdir is emulated by setting the working directory to the
	// absolute path that this file was opened at, which is why we have to
	// resolve and capture it here.
	if filePath[0] != '/' {
		wd, err := syscall.Getwd()
		if err != nil {
			return -1, poll.SysFile{}, err
		}
		absPath = joinPath(wd, filePath)
	}
	fd, err := syscall.Open(absPath, flag, perm)
	return fd, poll.SysFile{Path: absPath}, err
}

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows && !plan9
// +build !windows,!plan9

package robustio

import (
	"os"
	"syscall"
)

func getFileID(filename string) (FileID, error) {
	fi, err := os.Stat(filename)
	if err != nil {
		return FileID{}, err
	}
	stat := fi.Sys().(*syscall.Stat_t)
	return FileID{
		device: uint64(stat.Dev), // (int32 on darwin, uint64 on linux)
		inode:  stat.Ino,
	}, nil
}

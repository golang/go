// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows && !plan9
// +build !windows,!plan9

// TODO(adonovan): use 'unix' tag when go1.19 can be assumed.

package robustio

import (
	"os"
	"syscall"
	"time"
)

func getFileID(filename string) (FileID, time.Time, error) {
	fi, err := os.Stat(filename)
	if err != nil {
		return FileID{}, time.Time{}, err
	}
	stat := fi.Sys().(*syscall.Stat_t)
	return FileID{
		device: uint64(stat.Dev), // (int32 on darwin, uint64 on linux)
		inode:  stat.Ino,
	}, fi.ModTime(), nil
}

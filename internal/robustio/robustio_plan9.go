// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build plan9
// +build plan9

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
	dir := fi.Sys().(*syscall.Dir)
	return FileID{
		device: uint64(dir.Type)<<32 | uint64(dir.Dev),
		inode:  dir.Qid.Path,
	}, fi.ModTime(), nil
}

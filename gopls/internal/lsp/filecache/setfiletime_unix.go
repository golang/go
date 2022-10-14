// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !windows
// +build !windows

// TODO(adonovan): use 'unix' tag when we can rely on newer go command.

package filecache

import (
	"syscall"
	"time"
)

// setFileTime updates the access and modification times of a file.
//
// (Traditionally the access time would be updated automatically, but
// for efficiency most POSIX systems have for many years set the
// noatime mount option to avoid every open or read operation
// entailing a metadata write.)
func setFileTime(filename string, atime, mtime time.Time) error {
	return syscall.Utimes(filename, []syscall.Timeval{
		syscall.NsecToTimeval(atime.UnixNano()),
		syscall.NsecToTimeval(mtime.UnixNano()),
	})
}

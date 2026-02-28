// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package os

import "syscall"

func isErrNoFollow(err error) bool {
	return err == syscall.ELOOP
}

func newDirFile(fd syscall.Handle, name string) (*File, error) {
	return newFile(fd, name, kindOpenFile, false), nil
}

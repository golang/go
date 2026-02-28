// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build js

package fdtest

import (
	"syscall"
)

// Exists returns true if fd is a valid file descriptor.
func Exists(fd uintptr) bool {
	var s syscall.Stat_t
	err := syscall.Fstat(int(fd), &s)
	return err != syscall.EBADF
}

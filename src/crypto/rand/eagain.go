// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package rand

import (
	"os"
	"syscall"
)

func init() {
	isEAGAIN = unixIsEAGAIN
}

// unixIsEAGAIN reports whether err is a syscall.EAGAIN wrapped in a PathError.
// See golang.org/issue/9205
func unixIsEAGAIN(err error) bool {
	if pe, ok := err.(*os.PathError); ok {
		if errno, ok := pe.Err.(syscall.Errno); ok && errno == syscall.EAGAIN {
			return true
		}
	}
	return false
}

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || freebsd || linux || netbsd || openbsd

package toolchain

import (
	"io/fs"
	"syscall"
)

// sysWriteBits determines which bits to OR into the mode to make a directory writable.
// It must be called when there are no other file system operations happening.
func sysWriteBits() fs.FileMode {
	// Read current umask. There's no way to read it without also setting it,
	// so set it conservatively and then restore the original one.
	m := syscall.Umask(0o777)
	syscall.Umask(m)    // restore bits
	if m&0o22 == 0o22 { // group and world are unwritable by default
		return 0o700
	}
	if m&0o2 == 0o2 { // group is writable by default, but not world
		return 0o770
	}
	return 0o777 // everything is writable by default
}

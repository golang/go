// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

package modload

import (
	"os"
	"syscall"
)

// hasWritePerm reports whether the current user has permission to write to the
// file with the given info.
//
// Although the root user on most Unix systems can write to files even without
// permission, hasWritePerm reports false if no appropriate permission bit is
// set even if the current user is root.
func hasWritePerm(path string, fi os.FileInfo) bool {
	if os.Getuid() == 0 {
		// The root user can access any file, but we still want to default to
		// read-only mode if the go.mod file is marked as globally non-writable.
		// (If the user really intends not to be in readonly mode, they can
		// pass -mod=mod explicitly.)
		return fi.Mode()&0222 != 0
	}

	const W_OK = 0x2
	return syscall.Access(path, W_OK) == nil
}

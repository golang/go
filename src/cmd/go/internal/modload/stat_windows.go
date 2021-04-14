// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package modload

import "io/fs"

// hasWritePerm reports whether the current user has permission to write to the
// file with the given info.
func hasWritePerm(_ string, fi fs.FileInfo) bool {
	// Windows has a read-only attribute independent of ACLs, so use that to
	// determine whether the file is intended to be overwritten.
	//
	// Per https://golang.org/pkg/os/#Chmod:
	// “On Windows, only the 0200 bit (owner writable) of mode is used; it
	// controls whether the file's read-only attribute is set or cleared.”
	return fi.Mode()&0200 != 0
}

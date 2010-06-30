// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Windows environment variables.

package os

import (
	"syscall"
	"utf16"
)

func TempDir() string {
	const pathSep = '\\'
	dirw := make([]uint16, syscall.MAX_PATH)
	n, _ := syscall.GetTempPath(uint32(len(dirw)), &dirw[0])
	if n > uint32(len(dirw)) {
		dirw = make([]uint16, n)
		n, _ = syscall.GetTempPath(uint32(len(dirw)), &dirw[0])
		if n > uint32(len(dirw)) {
			n = 0
		}
	}
	if n > 0 && dirw[n-1] == pathSep {
		n--
	}
	return string(utf16.Decode(dirw[0:n]))
}

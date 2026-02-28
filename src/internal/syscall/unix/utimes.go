// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !wasip1

package unix

import (
	"syscall"
	_ "unsafe" // for //go:linkname
)

//go:linkname Utimensat syscall.utimensat
func Utimensat(dirfd int, path string, times *[2]syscall.Timespec, flag int) error

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || (openbsd && !mips64) || solaris

package poll

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

//go:linkname writev syscall.writev
func writev(fd int, iovecs []syscall.Iovec) (uintptr, error)

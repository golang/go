// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || (openbsd && !mips64) || solaris

package poll

import _ "unsafe" // for go:linkname

// Implemented in the syscall package.
//
//go:linkname fcntl syscall.fcntl
func fcntl(fd int, cmd int, arg int) (int, error)

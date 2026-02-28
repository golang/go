// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || solaris

package unix

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

func IsNonblock(fd int) (nonblocking bool, err error) {
	flag, e1 := fcntl(fd, syscall.F_GETFL, 0)
	if e1 != nil {
		return false, e1
	}
	return flag&syscall.O_NONBLOCK != 0, nil
}

// Implemented in the syscall package.
//
//go:linkname fcntl syscall.fcntl
func fcntl(fd int, cmd int, arg int) (int, error)

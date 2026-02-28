// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package unix

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

func IsNonblock(fd int) (nonblocking bool, err error) {
	flags, e1 := fd_fdstat_get_flags(fd)
	if e1 != nil {
		return false, e1
	}
	return flags&syscall.FDFLAG_NONBLOCK != 0, nil
}

func HasNonblockFlag(flag int) bool {
	return flag&syscall.FDFLAG_NONBLOCK != 0
}

// This helper is implemented in the syscall package. It means we don't have
// to redefine the fd_fdstat_get host import or the fdstat struct it
// populates.
//
//go:linkname fd_fdstat_get_flags syscall.fd_fdstat_get_flags
func fd_fdstat_get_flags(fd int) (uint32, error)

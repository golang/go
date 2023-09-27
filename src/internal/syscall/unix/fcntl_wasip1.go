// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package unix

import "syscall"

func Fcntl(fd int, cmd int, arg int) (int, error) {
	if cmd == syscall.F_GETFL {
		flags, err := fd_fdstat_get_flags(fd)
		return int(flags), err
	}
	return 0, syscall.ENOSYS
}

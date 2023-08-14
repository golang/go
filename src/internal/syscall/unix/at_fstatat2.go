// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd || (linux && loong64)

package unix

import "syscall"

func Fstatat(dirfd int, path string, stat *syscall.Stat_t, flags int) error {
	return syscall.Fstatat(dirfd, path, stat, flags)
}

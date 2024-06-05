// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix && !darwin && !dragonfly && !freebsd && !linux && !openbsd && !netbsd

package unix

import "syscall"

func Eaccess(path string, mode uint32) error {
	return syscall.ENOSYS
}

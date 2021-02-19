// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || freebsd || openbsd || netbsd
// +build darwin freebsd openbsd netbsd

package fastwalk

import "syscall"

func direntNamlen(dirent *syscall.Dirent) uint64 {
	return uint64(dirent.Namlen)
}

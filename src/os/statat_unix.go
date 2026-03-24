// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || wasip1 || linux || netbsd || openbsd || solaris

package os

import (
	"internal/syscall/unix"
)

func (f *File) lstatatNolog(name string) (FileInfo, error) {
	var fs fileStat
	if err := f.pfd.Fstatat(name, &fs.sys, unix.AT_SYMLINK_NOFOLLOW); err != nil {
		return nil, f.wrapErr("fstatat", err)
	}
	fillFileStatFromSys(&fs, name)
	return &fs, nil
}

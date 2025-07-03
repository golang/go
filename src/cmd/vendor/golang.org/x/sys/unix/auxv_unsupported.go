// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !go1.21 && (aix || darwin || dragonfly || freebsd || linux || netbsd || openbsd || solaris || zos)

package unix

import "syscall"

func Auxv() ([][2]uintptr, error) {
	return nil, syscall.ENOTSUP
}

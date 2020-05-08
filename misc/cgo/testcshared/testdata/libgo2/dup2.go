// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux,!arm64 netbsd openbsd

package main

import "syscall"

func dup2(oldfd, newfd int) error {
	return syscall.Dup2(oldfd, newfd)
}

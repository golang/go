// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux,arm64

package main

import "syscall"

func dup2(oldfd, newfd int) error {
	return syscall.Dup3(oldfd, newfd, 0)
}

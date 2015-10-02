// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build freebsd linux

package net

import "syscall"

var (
	// Placeholders for socket system calls.
	accept4Func func(int, int) (int, syscall.Sockaddr, error) = syscall.Accept4
)

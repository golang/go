// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || darwin || dragonfly || freebsd || (js && wasm) || linux || netbsd || openbsd || solaris

package net

import "syscall"

var (
	testHookDialChannel  = func() {} // for golang.org/issue/5349
	testHookCanceledDial = func() {} // for golang.org/issue/16523

	// Placeholders for socket system calls.
	socketFunc        func(int, int, int) (int, error)  = syscall.Socket
	connectFunc       func(int, syscall.Sockaddr) error = syscall.Connect
	listenFunc        func(int, int) error              = syscall.Listen
	getsockoptIntFunc func(int, int, int) (int, error)  = syscall.GetsockoptInt
)

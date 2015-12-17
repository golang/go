// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux nacl netbsd openbsd solaris

package net

import (
	"syscall"
	"time"
)

var (
	testHookDialChannel = func() {} // see golang.org/issue/5349
	testHookDNSDialer   = func(d time.Duration) dnsDialer { return &Dialer{Timeout: d} }

	// Placeholders for socket system calls.
	socketFunc        func(int, int, int) (int, error)         = syscall.Socket
	closeFunc         func(int) error                          = syscall.Close
	connectFunc       func(int, syscall.Sockaddr) error        = syscall.Connect
	listenFunc        func(int, int) error                     = syscall.Listen
	acceptFunc        func(int) (int, syscall.Sockaddr, error) = syscall.Accept
	getsockoptIntFunc func(int, int, int) (int, error)         = syscall.GetsockoptInt
)

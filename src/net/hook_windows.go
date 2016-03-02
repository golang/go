// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"
	"time"
)

var (
	testHookDialChannel = func() { time.Sleep(time.Millisecond) } // see golang.org/issue/5349

	// Placeholders for socket system calls.
	socketFunc    func(int, int, int) (syscall.Handle, error)                                                             = syscall.Socket
	closeFunc     func(syscall.Handle) error                                                                              = syscall.Closesocket
	connectFunc   func(syscall.Handle, syscall.Sockaddr) error                                                            = syscall.Connect
	connectExFunc func(syscall.Handle, syscall.Sockaddr, *byte, uint32, *uint32, *syscall.Overlapped) error               = syscall.ConnectEx
	listenFunc    func(syscall.Handle, int) error                                                                         = syscall.Listen
	acceptFunc    func(syscall.Handle, syscall.Handle, *byte, uint32, uint32, uint32, *uint32, *syscall.Overlapped) error = syscall.AcceptEx
)

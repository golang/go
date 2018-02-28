// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/syscall/windows"
	"syscall"
	"time"
)

var (
	testHookDialChannel = func() { time.Sleep(time.Millisecond) } // see golang.org/issue/5349

	// Placeholders for socket system calls.
	socketFunc    func(int, int, int) (syscall.Handle, error)                                                 = syscall.Socket
	wsaSocketFunc func(int32, int32, int32, *syscall.WSAProtocolInfo, uint32, uint32) (syscall.Handle, error) = windows.WSASocket
	connectFunc   func(syscall.Handle, syscall.Sockaddr) error                                                = syscall.Connect
	listenFunc    func(syscall.Handle, int) error                                                             = syscall.Listen
)

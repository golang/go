// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import "syscall"

// CloseFunc is used to hook the close call.
var CloseFunc func(syscall.Handle) error = syscall.Closesocket

// AcceptFunc is used to hook the accept call.
var AcceptFunc func(syscall.Handle, syscall.Handle, *byte, uint32, uint32, uint32, *uint32, *syscall.Overlapped) error = syscall.AcceptEx

// ConnectExFunc is used to hook the ConnectEx call.
var ConnectExFunc func(syscall.Handle, syscall.Sockaddr, *byte, uint32, *uint32, *syscall.Overlapped) error = syscall.ConnectEx

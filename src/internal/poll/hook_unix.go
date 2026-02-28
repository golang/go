// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || (js && wasm) || wasip1

package poll

import "syscall"

// CloseFunc is used to hook the close call.
var CloseFunc func(int) error = syscall.Close

// AcceptFunc is used to hook the accept call.
var AcceptFunc func(int) (int, syscall.Sockaddr, error) = syscall.Accept

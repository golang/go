// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !windows

package cgotest

import "syscall"

var syscall_dot_SIGCHLD = syscall.SIGCHLD

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build nacl solaris

package net

import "syscall"

func maxListenerBacklog() int {
	// TODO: Implement this
	// NOTE: Never return a number bigger than 1<<16 - 1. See issue 5030.
	return syscall.SOMAXCONN
}

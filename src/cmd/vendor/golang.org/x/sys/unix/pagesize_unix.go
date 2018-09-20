// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd solaris

// For Unix, get the pagesize from the runtime.

package unix

import "syscall"

func Getpagesize() int {
	return syscall.Getpagesize()
}

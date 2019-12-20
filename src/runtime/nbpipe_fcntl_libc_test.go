// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin solaris

package runtime_test

import (
	"runtime"
	"syscall"
)

// Call fcntl libc function rather than calling syscall.
func fcntl(fd uintptr, cmd int, arg uintptr) (uintptr, syscall.Errno) {
	res, errno := runtime.Fcntl(fd, uintptr(cmd), arg)
	return res, syscall.Errno(errno)
}

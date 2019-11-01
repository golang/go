// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	"runtime"
	"syscall"
)

// We can't call syscall.Syscall on AIX. Therefore, fcntl is exported from the
// runtime in export_aix_test.go.
func fcntl(fd uintptr, cmd int, arg uintptr) (uintptr, syscall.Errno) {
	res, errno := runtime.Fcntl(fd, uintptr(cmd), arg)
	return res, syscall.Errno(errno)
}

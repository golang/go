// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"syscall"
	_ "unsafe" // for go:linkname
)

// Fsync invokes SYS_FCNTL with SYS_FULLFSYNC because
// on OS X, SYS_FSYNC doesn't fully flush contents to disk.
// See Issue #26650 as well as the man page for fsync on OS X.
func (fd *FD) Fsync() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()

	_, e1 := fcntl(fd.Sysfd, syscall.F_FULLFSYNC, 0)
	return e1
}

// Implemented in syscall/syscall_darwin.go.
//go:linkname fcntl syscall.fcntl
func fcntl(fd int, cmd int, arg int) (int, error)

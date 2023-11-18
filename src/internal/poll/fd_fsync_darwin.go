// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"errors"
	"internal/syscall/unix"
	"syscall"
)

// Fsync invokes SYS_FCNTL with SYS_FULLFSYNC because
// on OS X, SYS_FSYNC doesn't fully flush contents to disk.
// See Issue #26650 as well as the man page for fsync on OS X.
func (fd *FD) Fsync() error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return ignoringEINTR(func() error {
		_, err := unix.Fcntl(fd.Sysfd, syscall.F_FULLFSYNC, 0)

		// There are scenarios such as SMB mounts where fcntl will fail
		// with ENOTSUP. In those cases fallback to fsync.
		// See #64215
		if err != nil && errors.Is(err, syscall.ENOTSUP) {
			err = syscall.Fsync(fd.Sysfd)
		}
		return err
	})
}

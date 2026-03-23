// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build linux

package unix

import (
	"internal/strconv"
	"syscall"
)

func Fchmodat(dirfd int, path string, mode uint32, flags int) error {
	// On Linux, the fchmodat syscall silently ignores the AT_SYMLINK_NOFOLLOW flag.
	// We need to use fchmodat2 instead.
	// syscall.Fchmodat handles this.
	if err := syscall.Fchmodat(dirfd, path, mode, flags); err != syscall.EOPNOTSUPP {
		return err
	}

	// This kernel doesn't appear to support fchmodat2 (added in Linux 6.6).
	// We can't fall back to Fchmod, because it requires write permissions on the file.
	// Instead, use the same workaround as GNU libc and musl, which is to open the file
	// and then fchmodat the FD in /proc/self/fd.
	// See: https://lwn.net/Articles/939217/
	fd, err := Openat(dirfd, path, O_PATH|syscall.O_NOFOLLOW|syscall.O_CLOEXEC, 0)
	if err != nil {
		return err
	}
	defer syscall.Close(fd)
	procPath := "/proc/self/fd/" + strconv.Itoa(fd)

	// Check to see if this file is a symlink.
	// (We passed O_NOFOLLOW above, but O_PATH|O_NOFOLLOW will open a symlink.)
	var st syscall.Stat_t
	if err := syscall.Stat(procPath, &st); err != nil {
		if err == syscall.ENOENT {
			// /proc has probably not been mounted. Give up.
			return syscall.EOPNOTSUPP
		}
		return err
	}
	if st.Mode&syscall.S_IFMT == syscall.S_IFLNK {
		// fchmodat on the proc FD for a symlink apparently gives inconsistent
		// results, so just refuse to try.
		return syscall.EOPNOTSUPP
	}

	return syscall.Fchmodat(AT_FDCWD, procPath, mode, flags&^AT_SYMLINK_NOFOLLOW)
}

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cache

import "syscall"

// statfs returns the blocks available to unprivileged users, the total blocks,
// and the block size of the file system holding dir, or zeros if unknown.
func statfs(dir string) (avail, total, bsize uint64) {
	var st syscall.Statfs_t
	if syscall.Statfs(dir, &st) != nil {
		return 0, 0, 0
	}
	// Block counts are in Frsize units, as in statvfs (Bsize if Frsize is unset).
	// Converting to int64 handles int32, uint32, and int64 fields across GOARCHes.
	size := int64(st.Frsize)
	if size <= 0 {
		size = int64(st.Bsize)
	}
	if size <= 0 {
		return 0, 0, 0
	}
	return st.Bavail, st.Blocks, uint64(size)
}

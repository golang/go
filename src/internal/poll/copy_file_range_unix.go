// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build freebsd || linux

package poll

import "internal/syscall/unix"

// CopyFileRange copies at most remain bytes of data from src to dst, using
// the copy_file_range system call. dst and src must refer to regular files.
func CopyFileRange(dst, src *FD, remain int64) (written int64, handled bool, err error) {
	if !supportCopyFileRange() {
		return 0, false, nil
	}

	for remain > 0 {
		max := remain
		if max > maxCopyFileRangeRound {
			max = maxCopyFileRangeRound
		}
		n, e := copyFileRange(dst, src, int(max))
		if n > 0 {
			remain -= n
			written += n
		}
		handled, err = handleCopyFileRangeErr(e, n, written)
		if n == 0 || !handled || err != nil {
			return
		}
	}

	return written, true, nil
}

// copyFileRange performs one round of copy_file_range(2).
func copyFileRange(dst, src *FD, max int) (written int64, err error) {
	// For Linux, the signature of copy_file_range(2) is:
	//
	// ssize_t copy_file_range(int fd_in, loff_t *off_in,
	//                         int fd_out, loff_t *off_out,
	//                         size_t len, unsigned int flags);
	//
	// For FreeBSD, the signature of copy_file_range(2) is:
	//
	// ssize_t
	// copy_file_range(int infd, off_t *inoffp, int outfd, off_t *outoffp,
	//                 size_t len, unsigned int flags);
	//
	// Note that in the call to unix.CopyFileRange below, we use nil
	// values for off_in/off_out and inoffp/outoffp, which means "the file
	// offset for infd(fd_in) or outfd(fd_out) respectively will be used and
	// updated by the number of bytes copied".
	//
	// That is why we must acquire locks for both file descriptors (and why
	// this whole machinery is in the internal/poll package to begin with).
	if err := dst.writeLock(); err != nil {
		return 0, err
	}
	defer dst.writeUnlock()
	if err := src.readLock(); err != nil {
		return 0, err
	}
	defer src.readUnlock()
	return ignoringEINTR2(func() (int64, error) {
		n, err := unix.CopyFileRange(src.Sysfd, nil, dst.Sysfd, nil, max, 0)
		return int64(n), err
	})
}

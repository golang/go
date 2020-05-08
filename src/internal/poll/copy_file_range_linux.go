// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"internal/syscall/unix"
	"sync/atomic"
	"syscall"
)

var copyFileRangeSupported int32 = 1 // accessed atomically

const maxCopyFileRangeRound = 1 << 30

// CopyFileRange copies at most remain bytes of data from src to dst, using
// the copy_file_range system call. dst and src must refer to regular files.
func CopyFileRange(dst, src *FD, remain int64) (written int64, handled bool, err error) {
	if atomic.LoadInt32(&copyFileRangeSupported) == 0 {
		return 0, false, nil
	}
	for remain > 0 {
		max := remain
		if max > maxCopyFileRangeRound {
			max = maxCopyFileRangeRound
		}
		n, err := copyFileRange(dst, src, int(max))
		switch err {
		case syscall.ENOSYS:
			// copy_file_range(2) was introduced in Linux 4.5.
			// Go supports Linux >= 2.6.33, so the system call
			// may not be present.
			//
			// If we see ENOSYS, we have certainly not transfered
			// any data, so we can tell the caller that we
			// couldn't handle the transfer and let them fall
			// back to more generic code.
			//
			// Seeing ENOSYS also means that we will not try to
			// use copy_file_range(2) again.
			atomic.StoreInt32(&copyFileRangeSupported, 0)
			return 0, false, nil
		case syscall.EXDEV, syscall.EINVAL:
			// Prior to Linux 5.3, it was not possible to
			// copy_file_range across file systems. Similarly to
			// the ENOSYS case above, if we see EXDEV, we have
			// not transfered any data, and we can let the caller
			// fall back to generic code.
			//
			// As for EINVAL, that is what we see if, for example,
			// dst or src refer to a pipe rather than a regular
			// file. This is another case where no data has been
			// transfered, so we consider it unhandled.
			return 0, false, nil
		case nil:
			if n == 0 {
				// src is at EOF, which means we are done.
				return written, true, nil
			}
			remain -= n
			written += n
		default:
			return written, true, err
		}
	}
	return written, true, nil
}

// copyFileRange performs one round of copy_file_range(2).
func copyFileRange(dst, src *FD, max int) (written int64, err error) {
	// The signature of copy_file_range(2) is:
	//
	// ssize_t copy_file_range(int fd_in, loff_t *off_in,
	//                         int fd_out, loff_t *off_out,
	//                         size_t len, unsigned int flags);
	//
	// Note that in the call to unix.CopyFileRange below, we use nil
	// values for off_in and off_out. For the system call, this means
	// "use and update the file offsets". That is why we must acquire
	// locks for both file descriptors (and why this whole machinery is
	// in the internal/poll package to begin with).
	if err := dst.writeLock(); err != nil {
		return 0, err
	}
	defer dst.writeUnlock()
	if err := src.readLock(); err != nil {
		return 0, err
	}
	defer src.readUnlock()
	var n int
	for {
		n, err = unix.CopyFileRange(src.Sysfd, nil, dst.Sysfd, nil, max, 0)
		if err != syscall.EINTR {
			break
		}
	}
	return int64(n), err
}

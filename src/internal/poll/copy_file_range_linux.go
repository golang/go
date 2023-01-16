// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"internal/syscall/unix"
	"sync"
	"syscall"
)

var (
	kernelVersion53Once sync.Once
	kernelVersion53     bool
)

const maxCopyFileRangeRound = 1 << 30

// CopyFileRange copies at most remain bytes of data from src to dst, using
// the copy_file_range system call. dst and src must refer to regular files.
func CopyFileRange(dst, src *FD, remain int64) (written int64, handled bool, err error) {
	kernelVersion53Once.Do(func() {
		major, minor := unix.KernelVersion()
		// copy_file_range(2) is broken in various ways on kernels older than 5.3,
		// see issue #42400 and
		// https://man7.org/linux/man-pages/man2/copy_file_range.2.html#VERSIONS
		if major > 5 || (major == 5 && minor >= 3) {
			kernelVersion53 = true
		}
	})

	if !kernelVersion53 {
		return 0, false, nil
	}

	for remain > 0 {
		max := remain
		if max > maxCopyFileRangeRound {
			max = maxCopyFileRangeRound
		}
		n, err := copyFileRange(dst, src, int(max))
		switch err {
		case syscall.EXDEV, syscall.EINVAL, syscall.EIO, syscall.EOPNOTSUPP, syscall.EPERM:
			// Prior to Linux 5.3, it was not possible to
			// copy_file_range across file systems. An attempt
			// to do this will result in a EXDEV error.
			//
			// Even though we have checked the kernel version and blocked
			// the attempts to copy_file_range(2) when the kernel version
			// is older than 5.3, but until now the latest kernel (5.19.x)
			// may still return EXDEV error in certain cases.
			//
			// If we see EXDEV, we have not transferred any data,
			// and we can let the caller fall back to generic code.
			//
			// As for EINVAL, that is what we see if, for example,
			// dst or src refer to a pipe rather than a regular
			// file. This is another case where no data has been
			// transferred, so we consider it unhandled.
			//
			// If src and dst are on CIFS, we can see EIO.
			// See issue #42334.
			//
			// If the file is on NFS, we can see EOPNOTSUPP.
			// See issue #40731.
			//
			// If the process is running inside a Docker container,
			// we might see EPERM instead of ENOSYS. See issue
			// #40893. Since EPERM might also be a legitimate error,
			// don't mark copy_file_range(2) as unsupported.
			return 0, false, nil
		case nil:
			if n == 0 {
				// If we did not read any bytes at all,
				// then this file may be in a file system
				// where copy_file_range silently fails.
				// https://lore.kernel.org/linux-fsdevel/20210126233840.GG4626@dread.disaster.area/T/#m05753578c7f7882f6e9ffe01f981bc223edef2b0
				if written == 0 {
					return 0, false, nil
				}
				// Otherwise src is at EOF, which means
				// we are done.
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

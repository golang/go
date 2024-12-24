// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"internal/syscall/unix"
	"sync"
	"syscall"
)

func supportCopyFileRange() bool {
	return isKernelVersionGE53()
}

var isKernelVersionGE53 = sync.OnceValue(func() bool {
	major, minor := unix.KernelVersion()
	// copy_file_range(2) is broken in various ways on kernels older than 5.3,
	// see https://go.dev/issue/42400 and
	// https://man7.org/linux/man-pages/man2/copy_file_range.2.html#VERSIONS
	return major > 5 || (major == 5 && minor >= 3)
})

// For best performance, call copy_file_range() with the largest len value
// possible. Linux sets up a limitation of data transfer for most of its I/O
// system calls, as MAX_RW_COUNT (INT_MAX & PAGE_MASK). This value equals to
// the maximum integer value minus a page size that is typically 2^12=4096 bytes.
// That is to say, it's the maximum integer value with the lowest 12 bits unset,
// which is 0x7ffff000.
const maxCopyFileRangeRound = 0x7ffff000

func handleCopyFileRangeErr(err error, copied, written int64) (bool, error) {
	switch err {
	case syscall.ENOSYS:
		// copy_file_range(2) was introduced in Linux 4.5.
		// Go supports Linux >= 3.2, so the system call
		// may not be present.
		//
		// If we see ENOSYS, we have certainly not transferred
		// any data, so we can tell the caller that we
		// couldn't handle the transfer and let them fall
		// back to more generic code.
		return false, nil
	case syscall.EXDEV, syscall.EINVAL, syscall.EIO, syscall.EOPNOTSUPP, syscall.EPERM:
		// Prior to Linux 5.3, it was not possible to
		// copy_file_range across file systems. Similarly to
		// the ENOSYS case above, if we see EXDEV, we have
		// not transferred any data, and we can let the caller
		// fall back to generic code.
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
		return false, nil
	case nil:
		if copied == 0 {
			// If we did not read any bytes at all,
			// then this file may be in a file system
			// where copy_file_range silently fails.
			// https://lore.kernel.org/linux-fsdevel/20210126233840.GG4626@dread.disaster.area/T/#m05753578c7f7882f6e9ffe01f981bc223edef2b0
			if written == 0 {
				return false, nil
			}

			// Otherwise src is at EOF, which means
			// we are done.
		}
	}
	return true, err
}

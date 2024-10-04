// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"internal/syscall/unix"
	"syscall"
)

func supportCopyFileRange() bool {
	return unix.SupportCopyFileRange()
}

// For best performance, call copy_file_range() with the largest len value
// possible. It is interruptible on most file systems, so there is no penalty
// for using very large len values, even SSIZE_MAX.
const maxCopyFileRangeRound = 1<<31 - 1

func handleCopyFileRangeErr(err error, copied, written int64) (bool, error) {
	switch err {
	case syscall.ENOSYS:
		// The copy_file_range(2) function first appeared in FreeBSD 13.0.
		// Go supports FreeBSD >= 12, so the system call
		// may not be present. We've detected the FreeBSD version with
		// unix.SupportCopyFileRange() at the beginning of this function,
		// but we still want to check for ENOSYS here to prevent some rare
		// case like https://go.dev/issue/58592
		//
		// If we see ENOSYS, we have certainly not transferred
		// any data, so we can tell the caller that we
		// couldn't handle the transfer and let them fall
		// back to more generic code.
		return false, nil
	case syscall.EFBIG, syscall.EINVAL, syscall.EIO:
		// For EFBIG, the copy has exceeds the process's file size limit
		// or the maximum file size for the filesystem dst resides on, in
		// this case, we leave it to generic copy.
		//
		// For EINVAL, there could be a few reasons:
		// 1. Either dst or src refers to a file object that
		// is not a regular file, for instance, a pipe.
		// 2. src and dst refer to the same file and byte ranges
		// overlap.
		// 3. The flags argument is not 0.
		// Neither of these cases should be considered handled by
		// copy_file_range(2) because there is no data transfer, so
		// just fall back to generic copy.
		return false, nil
	}
	return true, err
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd

package poll

import "syscall"

// maxSendfileSize is the largest chunk size we ask the kernel to copy
// at a time.
// sendfile(2)s on *BSD and Darwin don't have a limit on the size of
// data to copy at a time, we pick the typical SSIZE_MAX on 32-bit systems,
// which ought to be sufficient for all practical purposes.
const maxSendfileSize int = 1<<31 - 1

// SendFile wraps the sendfile system call.
func SendFile(dstFD *FD, src int, pos, remain int64) (written int64, err error, handled bool) {
	defer func() {
		TestHookDidSendFile(dstFD, src, written, err, handled)
	}()
	if err := dstFD.writeLock(); err != nil {
		return 0, err, false
	}
	defer dstFD.writeUnlock()

	if err := dstFD.pd.prepareWrite(dstFD.isFile); err != nil {
		return 0, err, false
	}

	dst := dstFD.Sysfd
	for remain > 0 {
		n := maxSendfileSize
		if int64(n) > remain {
			n = int(remain)
		}
		pos1 := pos
		n, err = syscall.Sendfile(dst, src, &pos1, n)
		if n > 0 {
			pos += int64(n)
			written += int64(n)
			remain -= int64(n)
		}
		if err == syscall.EINTR {
			continue
		}
		// This includes syscall.ENOSYS (no kernel
		// support) and syscall.EINVAL (fd types which
		// don't implement sendfile), and other errors.
		// We should end the loop when there is no error
		// returned from sendfile(2) or it is not a retryable error.
		if err != syscall.EAGAIN {
			break
		}
		if err = dstFD.pd.waitWrite(dstFD.isFile); err != nil {
			break
		}
	}
	handled = written != 0 || (err != syscall.ENOSYS && err != syscall.EINVAL)
	return
}

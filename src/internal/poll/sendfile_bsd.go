// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd

package poll

import "syscall"

// maxSendfileSize is the largest chunk size we ask the kernel to copy
// at a time.
const maxSendfileSize int = 4 << 20

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
		m := n
		pos1 := pos
		n, err = syscall.Sendfile(dst, src, &pos1, n)
		if n > 0 {
			pos += int64(n)
			written += int64(n)
			remain -= int64(n)
			// (n, nil) indicates that sendfile(2) has transferred
			// the exact number of bytes we requested, or some unretryable
			// error have occurred with partial bytes sent. Either way, we
			// don't need to go through the following logic to check EINTR
			// or fell into dstFD.pd.waitWrite, just continue to send the
			// next chunk or break the loop.
			if n == m {
				continue
			} else if err != syscall.EAGAIN &&
				err != syscall.EINTR &&
				err != syscall.EBUSY {
				// Particularly, EPIPE. Errors like that would normally lead
				// the subsequent sendfile(2) call to (-1, EBADF).
				break
			}
		} else if err != syscall.EAGAIN && err != syscall.EINTR {
			// This includes syscall.ENOSYS (no kernel
			// support) and syscall.EINVAL (fd types which
			// don't implement sendfile), and other errors.
			// We should end the loop when there is no error
			// returned from sendfile(2) or it is not a retryable error.
			break
		}
		if err == syscall.EINTR {
			continue
		}
		if err = dstFD.pd.waitWrite(dstFD.isFile); err != nil {
			break
		}
	}
	if err == syscall.EAGAIN {
		err = nil
	}
	handled = written != 0 || (err != syscall.ENOSYS && err != syscall.EINVAL && err != syscall.EOPNOTSUPP && err != syscall.ENOTSUP)
	return
}

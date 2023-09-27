// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import "syscall"

// maxSendfileSize is the largest chunk size we ask the kernel to copy
// at a time.
const maxSendfileSize int = 4 << 20

// SendFile wraps the sendfile system call.
func SendFile(dstFD *FD, src int, remain int64) (int64, error, bool) {
	if err := dstFD.writeLock(); err != nil {
		return 0, err, false
	}
	defer dstFD.writeUnlock()
	if err := dstFD.pd.prepareWrite(dstFD.isFile); err != nil {
		return 0, err, false
	}

	dst := dstFD.Sysfd
	var (
		written int64
		err     error
		handled = true
	)
	for remain > 0 {
		n := maxSendfileSize
		if int64(n) > remain {
			n = int(remain)
		}
		n, err1 := syscall.Sendfile(dst, src, nil, n)
		if n > 0 {
			written += int64(n)
			remain -= int64(n)
		} else if n == 0 && err1 == nil {
			break
		}
		if err1 == syscall.EINTR {
			continue
		}
		if err1 == syscall.EAGAIN {
			if err1 = dstFD.pd.waitWrite(dstFD.isFile); err1 == nil {
				continue
			}
		}
		if err1 != nil {
			// This includes syscall.ENOSYS (no kernel
			// support) and syscall.EINVAL (fd types which
			// don't implement sendfile)
			err = err1
			handled = false
			break
		}
	}
	return written, err, handled
}

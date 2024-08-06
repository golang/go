// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import "syscall"

//go:cgo_ldflag "-lsendfile"

// Not strictly needed, but very helpful for debugging, see issue #10221.
//
//go:cgo_import_dynamic _ _ "libsendfile.so"
//go:cgo_import_dynamic _ _ "libsocket.so"

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
		pos1 := pos
		n, err = syscall.Sendfile(dst, src, &pos1, n)
		if err == syscall.EAGAIN || err == syscall.EINTR || err == syscall.EINVAL {
			// Partial write or other quirks may have occurred.
			//
			// For EINVAL, this is another quirk on SunOS: sendfile() claims to support
			// out_fd as a regular file but returns EINVAL when the out_fd is not a
			// socket of SOCK_STREAM, while it actually sends out data anyway and updates
			// the file offset.
			n = int(pos1 - pos)
		}
		if n > 0 {
			pos += int64(n)
			written += int64(n)
			remain -= int64(n)
			continue
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
	handled = written != 0 || (err != syscall.ENOSYS && err != syscall.EINVAL)
	return
}

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || linux || solaris

package poll

import (
	"io"
	"runtime"
	"syscall"
)

// SendFile wraps the sendfile system call.
//
// It copies data from src (a file descriptor) to dstFD,
// starting at the current position of src.
// It updates the current position of src to after the
// copied data.
//
// If size is zero, it copies the rest of src.
// Otherwise, it copies up to size bytes.
//
// The handled return parameter indicates whether SendFile
// was able to handle some or all of the operation.
// If handled is false, sendfile was unable to perform the copy,
// has not modified the source or destination,
// and the caller should perform the copy using a fallback implementation.
func SendFile(dstFD *FD, src int, size int64) (n int64, err error, handled bool) {
	if runtime.GOOS == "linux" {
		// Linux's sendfile doesn't require any setup:
		// It sends from the current position of the source file and
		// updates the position of the source after sending.
		return sendFile(dstFD, src, nil, size)
	}

	// Non-Linux sendfile implementations don't use the current position of the source file,
	// so we need to look up the position, pass it explicitly, and adjust it after
	// sendfile returns.
	start, err := ignoringEINTR2(func() (int64, error) {
		return syscall.Seek(src, 0, io.SeekCurrent)
	})
	if err != nil {
		return 0, err, false
	}

	pos := start
	n, err, handled = sendFile(dstFD, src, &pos, size)
	if n > 0 {
		ignoringEINTR2(func() (int64, error) {
			return syscall.Seek(src, start+n, io.SeekStart)
		})
	}
	return n, err, handled
}

// sendFile wraps the sendfile system call.
func sendFile(dstFD *FD, src int, offset *int64, size int64) (written int64, err error, handled bool) {
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
	for {
		// Some platforms support passing 0 to read to the end of the source,
		// but all platforms support just writing a large value.
		//
		// Limit the maximum size to fit in an int32, to avoid any possible overflow.
		chunk := 1<<31 - 1
		if size > 0 {
			chunk = int(min(size-written, int64(chunk)))
		}
		var n int
		n, err = sendFileChunk(dst, src, offset, chunk)
		if n > 0 {
			written += int64(n)
		}

		switch runtime.GOOS {
		case "solaris", "illumos":
			// A quirk on Solaris/illumos: sendfile() claims to support out_fd
			// as a regular file but returns EINVAL when the out_fd
			// is not a socket of SOCK_STREAM, while it actually sends
			// out data anyway and updates the file offset.
			//
			// We ignore EINVAL if any sendfile call returned n > 0,
			// to handle the case where the last call returns 0 to indicate
			// no more data to send.
			if err == syscall.EINVAL && written > 0 {
				err = nil
			}
		}

		switch err {
		case nil:
			// We're done if sendfile copied no bytes
			// (we're at the end of the source)
			// or if we have a size limit and have reached it.
			//
			// If sendfile copied some bytes and we don't have a size limit,
			// try again to see if there is more data to copy.
			if n == 0 || (size > 0 && written >= size) {
				return written, nil, true
			}
		case syscall.EAGAIN:
			// Darwin can return EAGAIN with n > 0,
			// so check to see if the write has completed.
			// So far as we know all other platforms only return EAGAIN when n == 0,
			// but checking is harmless.
			if size > 0 && written >= size {
				return written, nil, true
			}
			if err = dstFD.pd.waitWrite(dstFD.isFile); err != nil {
				return written, err, true
			}
		case syscall.EINTR:
			// Ignore.
		case syscall.ENOSYS, syscall.EINVAL, syscall.EOPNOTSUPP:
			// ENOSYS indicates no kernel support for sendfile.
			// EINVAL indicates a FD type which does not support sendfile.
			//
			// On Linux, copy_file_range can return EOPNOTSUPP when copying
			// to a NFS file (issue #40731); check for it here just in case.
			return written, err, written > 0
		default:
			// Not a retryable error.
			return written, err, true
		}
	}
}

func sendFileChunk(dst, src int, offset *int64, size int) (n int, err error) {
	switch runtime.GOOS {
	case "linux":
		// The offset is always nil on Linux.
		n, err = syscall.Sendfile(dst, src, offset, size)
	case "solaris", "illumos":
		// Trust the offset, not the return value from sendfile.
		start := *offset
		n, err = syscall.Sendfile(dst, src, offset, size)
		n = int(*offset - start)
	default:
		start := *offset
		n, err = syscall.Sendfile(dst, src, offset, size)
		if n > 0 {
			// The BSD implementations of syscall.Sendfile don't
			// update the offset parameter (despite it being a *int64).
			//
			// Trust the return value from sendfile, not the offset.
			*offset = start + int64(n)
		}
	}
	return
}

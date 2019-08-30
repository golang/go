// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/poll"
	"io"
	"os"
	"syscall"
)

// sendFile copies the contents of r to c using the TransmitFile
// system call to minimize copies.
//
// if handled == true, sendFile returns the number of bytes copied and any
// non-EOF error.
//
// if handled == false, sendFile performed no work.
func sendFile(fd *netFD, r io.Reader) (written int64, err error, handled bool) {
	var n int64 = 0 // by default, copy until EOF.

	lr, ok := r.(*io.LimitedReader)
	if ok {
		n, r = lr.N, lr.R
		if n <= 0 {
			return 0, nil, true
		}
	}

	f, ok := r.(*os.File)
	if !ok {
		return 0, nil, false
	}

	// TransmitFile can be invoked in one call with at most
	// 2,147,483,646 bytes: the maximum value for a 32-bit integer minus 1.
	// See https://docs.microsoft.com/en-us/windows/win32/api/mswsock/nf-mswsock-transmitfile
	const maxChunkSizePerCall = int64(0x7fffffff - 1)

	switch {
	case n <= maxChunkSizePerCall:
		// The file is within sendfile's limits.
		written, err = doSendFile(fd, lr, f, n)

	default:
		// Now invoke doSendFile on the file in chunks of upto 2GiB per chunk.
		for lr.N > 0 { // lr.N is decremented in every successful invocation of doSendFile.
			chunkSize := maxChunkSizePerCall
			if chunkSize > lr.N {
				chunkSize = lr.N
			}
			var nw int64
			nw, err = doSendFile(fd, lr, f, chunkSize)
			if err != nil {
				break
			}
			written += nw
		}
	}

	// If any byte was copied, regardless of any error
	// encountered mid-way, handled must be set to true.
	return written, err, written > 0
}

// doSendFile is a helper to invoke poll.SendFile.
// It will decrement lr.N by the number of written bytes.
func doSendFile(fd *netFD, lr *io.LimitedReader, f *os.File, remain int64) (written int64, err error) {
	done, err := poll.SendFile(&fd.pfd, syscall.Handle(f.Fd()), remain)
	if err != nil {
		return 0, wrapSyscallError("transmitfile", err)
	}
	if lr != nil {
		lr.N -= int64(done)
	}
	return int64(done), nil
}

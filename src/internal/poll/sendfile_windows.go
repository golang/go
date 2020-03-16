// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"io"
	"syscall"
)

// SendFile wraps the TransmitFile call.
func SendFile(fd *FD, src syscall.Handle, n int64) (written int64, err error) {
	if fd.kind == kindPipe {
		// TransmitFile does not work with pipes
		return 0, syscall.ESPIPE
	}

	if err := fd.writeLock(); err != nil {
		return 0, err
	}
	defer fd.writeUnlock()

	o := &fd.wop
	o.handle = src

	// TODO(brainman): skip calling syscall.Seek if OS allows it
	curpos, err := syscall.Seek(o.handle, 0, io.SeekCurrent)
	if err != nil {
		return 0, err
	}

	if n <= 0 { // We don't know the size of the file so infer it.
		// Find the number of bytes offset from curpos until the end of the file.
		n, err = syscall.Seek(o.handle, -curpos, io.SeekEnd)
		if err != nil {
			return
		}
		// Now seek back to the original position.
		if _, err = syscall.Seek(o.handle, curpos, io.SeekStart); err != nil {
			return
		}
	}

	// TransmitFile can be invoked in one call with at most
	// 2,147,483,646 bytes: the maximum value for a 32-bit integer minus 1.
	// See https://docs.microsoft.com/en-us/windows/win32/api/mswsock/nf-mswsock-transmitfile
	const maxChunkSizePerCall = int64(0x7fffffff - 1)

	for n > 0 {
		chunkSize := maxChunkSizePerCall
		if chunkSize > n {
			chunkSize = n
		}

		o.qty = uint32(chunkSize)
		o.o.Offset = uint32(curpos)
		o.o.OffsetHigh = uint32(curpos >> 32)

		nw, err := wsrv.ExecIO(o, func(o *operation) error {
			return syscall.TransmitFile(o.fd.Sysfd, o.handle, o.qty, 0, &o.o, nil, syscall.TF_WRITE_BEHIND)
		})
		if err != nil {
			return written, err
		}

		curpos += int64(nw)

		// Some versions of Windows (Windows 10 1803) do not set
		// file position after TransmitFile completes.
		// So just use Seek to set file position.
		if _, err = syscall.Seek(o.handle, curpos, io.SeekStart); err != nil {
			return written, err
		}

		n -= int64(nw)
		written += int64(nw)
	}

	return
}

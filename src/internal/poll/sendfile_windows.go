// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package poll

import (
	"io"
	"syscall"
)

// SendFile wraps the TransmitFile call.
func SendFile(fd *FD, src uintptr, size int64) (written int64, err error, handled bool) {
	defer func() {
		TestHookDidSendFile(fd, 0, written, err, written > 0)
	}()
	if fd.kind == kindPipe {
		// TransmitFile does not work with pipes
		return 0, syscall.ESPIPE, false
	}
	hsrc := syscall.Handle(src)
	if ft, _ := syscall.GetFileType(hsrc); ft == syscall.FILE_TYPE_PIPE {
		return 0, syscall.ESPIPE, false
	}

	if err := fd.writeLock(); err != nil {
		return 0, err, false
	}
	defer fd.writeUnlock()

	// Get the file size so we don't read past the end of the file.
	var fi syscall.ByHandleFileInformation
	if err := syscall.GetFileInformationByHandle(hsrc, &fi); err != nil {
		return 0, err, false
	}
	fileSize := int64(fi.FileSizeHigh)<<32 + int64(fi.FileSizeLow)
	startpos, err := syscall.Seek(hsrc, 0, io.SeekCurrent)
	if err != nil {
		return 0, err, false
	}
	maxSize := fileSize - startpos
	if size <= 0 {
		size = maxSize
	} else {
		size = min(size, maxSize)
	}

	defer func() {
		if written > 0 {
			// Some versions of Windows (Windows 10 1803) do not set
			// file position after TransmitFile completes.
			// So just use Seek to set file position.
			_, serr := syscall.Seek(hsrc, startpos+written, io.SeekStart)
			if err != nil {
				err = serr
			}
		}
	}()

	// TransmitFile can be invoked in one call with at most
	// 2,147,483,646 bytes: the maximum value for a 32-bit integer minus 1.
	// See https://docs.microsoft.com/en-us/windows/win32/api/mswsock/nf-mswsock-transmitfile
	const maxChunkSizePerCall = int64(0x7fffffff - 1)

	for size > 0 {
		chunkSize := maxChunkSizePerCall
		if chunkSize > size {
			chunkSize = size
		}

		fd.setOffset(startpos + written)
		n, err := fd.execIO('w', func(o *operation) (uint32, error) {
			err := syscall.TransmitFile(fd.Sysfd, hsrc, uint32(chunkSize), 0, &o.o, nil, syscall.TF_WRITE_BEHIND)
			if err != nil {
				return 0, err
			}
			return uint32(chunkSize), nil
		})
		if err != nil {
			return written, err, written > 0
		}

		size -= int64(n)
		written += int64(n)
	}

	return written, nil, written > 0
}

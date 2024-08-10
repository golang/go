// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/poll"
	"io"
)

var pollCopyFileRange = poll.CopyFileRange

func (f *File) writeTo(w io.Writer) (written int64, handled bool, err error) {
	return 0, false, nil
}

func (f *File) readFrom(r io.Reader) (written int64, handled bool, err error) {
	// copy_file_range(2) doesn't supports destinations opened with
	// O_APPEND, so don't bother to try zero-copy with these system calls.
	//
	// Visit https://man.freebsd.org/cgi/man.cgi?copy_file_range(2)#ERRORS for details.
	if f.appendMode {
		return 0, false, nil
	}

	var (
		remain int64
		lr     *io.LimitedReader
	)
	if lr, r, remain = tryLimitedReader(r); remain <= 0 {
		return 0, true, nil
	}

	var src *File
	switch v := r.(type) {
	case *File:
		src = v
	case fileWithoutWriteTo:
		src = v.File
	default:
		return 0, false, nil
	}

	if src.checkValid("ReadFrom") != nil {
		// Avoid returning the error as we report handled as false,
		// leave further error handling as the responsibility of the caller.
		return 0, false, nil
	}

	written, handled, err = pollCopyFileRange(&f.pfd, &src.pfd, remain)
	if lr != nil {
		lr.N -= written
	}

	return written, handled, wrapSyscallError("copy_file_range", err)
}

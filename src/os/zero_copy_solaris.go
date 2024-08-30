// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package os

import (
	"internal/poll"
	"io"
	"runtime"
	"syscall"
)

func (f *File) writeTo(w io.Writer) (written int64, handled bool, err error) {
	return 0, false, nil
}

// readFrom is basically a refactor of net.sendFile, but adapted to work for the target of *File.
func (f *File) readFrom(r io.Reader) (written int64, handled bool, err error) {
	// SunOS uses 0 as the "until EOF" value.
	// If you pass in more bytes than the file contains, it will
	// loop back to the beginning ad nauseam until it's sent
	// exactly the number of bytes told to. As such, we need to
	// know exactly how many bytes to send.
	var remain int64 = 0

	lr, ok := r.(*io.LimitedReader)
	if ok {
		remain, r = lr.N, lr.R
		if remain <= 0 {
			return 0, true, nil
		}
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

	// If fd_in and fd_out refer to the same file and the source and target ranges overlap,
	// sendfile(2) on SunOS will allow this kind of overlapping and work like a memmove,
	// in this case the file content remains the same after copying, which is not what we want.
	// Thus, we just bail out here and leave it to generic copy when it's a file copying itself.
	if f.pfd.Sysfd == src.pfd.Sysfd {
		return 0, false, nil
	}

	// sendfile() on illumos seems to incur intermittent failures when the
	// target file is a standard stream (stdout/stderr), we hereby skip any
	// anything other than regular files conservatively and leave them to generic copy.
	// Check out https://go.dev/issue/68863 for more details.
	if runtime.GOOS == "illumos" {
		fi, err := f.Stat()
		if err != nil {
			return 0, false, nil
		}
		st, ok := fi.Sys().(*syscall.Stat_t)
		if !ok {
			return 0, false, nil
		}
		if typ := st.Mode & syscall.S_IFMT; typ != syscall.S_IFREG {
			return 0, false, nil
		}
	}

	if remain == 0 {
		fi, err := src.Stat()
		if err != nil {
			return 0, false, err
		}

		remain = fi.Size()
	}

	// The other quirk with SunOS' sendfile implementation
	// is that it doesn't use the current position of the file
	// -- if you pass it offset 0, it starts from offset 0.
	// There's no way to tell it "start from current position",
	// so we have to manage that explicitly.
	pos, err := src.Seek(0, io.SeekCurrent)
	if err != nil {
		return
	}

	sc, err := src.SyscallConn()
	if err != nil {
		return
	}

	// System call sendfile()s on Solaris and illumos support file-to-file copying.
	// Check out https://docs.oracle.com/cd/E86824_01/html/E54768/sendfile-3ext.html and
	// https://docs.oracle.com/cd/E88353_01/html/E37843/sendfile-3c.html and
	// https://illumos.org/man/3EXT/sendfile for more details.
	rerr := sc.Read(func(fd uintptr) bool {
		written, err, handled = poll.SendFile(&f.pfd, int(fd), pos, remain)
		return true
	})
	if lr != nil {
		lr.N = remain - written
	}

	// This is another quirk on SunOS: sendfile() claims to support
	// out_fd as a regular file but returns EINVAL when the out_fd is not a
	// socket of SOCK_STREAM, while it actually sends out data anyway and updates
	// the file offset. In this case, we can just ignore the error.
	if err == syscall.EINVAL && written > 0 {
		err = nil
	}
	if err == nil {
		err = rerr
	}

	_, err1 := src.Seek(written, io.SeekCurrent)
	if err1 != nil && err == nil {
		return written, handled, err1
	}

	return written, handled, wrapSyscallError("sendfile", err)
}

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || js || wasip1 || windows

package os

import (
	"io"
	"syscall"
)

// wrapSyscallError takes an error and a syscall name. If the error is
// a syscall.Errno, it wraps it in an os.SyscallError using the syscall name.
func wrapSyscallError(name string, err error) error {
	if _, ok := err.(syscall.Errno); ok {
		err = NewSyscallError(name, err)
	}
	return err
}

// tryLimitedReader tries to assert the io.Reader to io.LimitedReader, it returns the io.LimitedReader,
// the underlying io.Reader and the remaining amount of bytes if the assertion succeeds,
// otherwise it just returns the original io.Reader and the theoretical unlimited remaining amount of bytes.
func tryLimitedReader(r io.Reader) (*io.LimitedReader, io.Reader, int64) {
	var remain int64 = 1<<63 - 1 // by default, copy until EOF

	lr, ok := r.(*io.LimitedReader)
	if !ok {
		return nil, r, remain
	}

	remain = lr.N
	return lr, lr.R, remain
}

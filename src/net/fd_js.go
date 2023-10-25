// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Fake networking for js/wasm. It is intended to allow tests of other package to pass.

//go:build js

package net

import (
	"os"
	"syscall"
)

func (fd *netFD) closeRead() error {
	if fd.fakeNetFD != nil {
		return fd.fakeNetFD.closeRead()
	}
	return os.NewSyscallError("closeRead", syscall.ENOTSUP)
}

func (fd *netFD) closeWrite() error {
	if fd.fakeNetFD != nil {
		return fd.fakeNetFD.closeWrite()
	}
	return os.NewSyscallError("closeRead", syscall.ENOTSUP)
}

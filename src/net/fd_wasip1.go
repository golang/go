// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build wasip1

package net

import (
	"syscall"
)

func (fd *netFD) closeRead() error {
	if fd.fakeNetFD != nil {
		return fd.fakeNetFD.closeRead()
	}
	return fd.shutdown(syscall.SHUT_RD)
}

func (fd *netFD) closeWrite() error {
	if fd.fakeNetFD != nil {
		return fd.fakeNetFD.closeWrite()
	}
	return fd.shutdown(syscall.SHUT_WR)
}

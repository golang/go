// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/syscall/windows"
	"os"
	"syscall"
)

const _SO_TYPE = windows.SO_TYPE

func dupSocket(h syscall.Handle) (syscall.Handle, error) {
	var info syscall.WSAProtocolInfo
	err := windows.WSADuplicateSocket(h, uint32(syscall.Getpid()), &info)
	if err != nil {
		return 0, err
	}
	return windows.WSASocket(-1, -1, -1, &info, 0, windows.WSA_FLAG_OVERLAPPED|windows.WSA_FLAG_NO_HANDLE_INHERIT)
}

func dupFileSocket(f *os.File) (syscall.Handle, error) {
	// The resulting handle should not be associated to an IOCP, else the IO operations
	// will block an OS thread, and that's not what net package users expect.
	h, err := dupSocket(syscall.Handle(f.Fd()))
	if err != nil {
		return 0, err
	}
	return h, nil
}

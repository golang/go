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
	// Call Fd to disassociate the IOCP from the handle,
	// it is not safe to share a duplicated handle
	// that is associated with IOCP.
	// Don't use the returned fd, as it might be closed
	// if f happens to be the last reference to the file.
	f.Fd()

	sc, err := f.SyscallConn()
	if err != nil {
		return 0, err
	}

	var h syscall.Handle
	var syserr error
	err = sc.Control(func(fd uintptr) {
		h, syserr = dupSocket(syscall.Handle(fd))
	})
	if err != nil {
		err = syserr
	}
	if err != nil {
		return 0, err
	}
	return h, nil
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/syscall/windows"
	"os"
	"syscall"
)

func maxListenerBacklog() int {
	// TODO: Implement this
	// NOTE: Never return a number bigger than 1<<16 - 1. See issue 5030.
	return syscall.SOMAXCONN
}

func sysSocket(family, sotype, proto int) (syscall.Handle, error) {
	s, err := wsaSocketFunc(int32(family), int32(sotype), int32(proto),
		nil, 0, windows.WSA_FLAG_OVERLAPPED|windows.WSA_FLAG_NO_HANDLE_INHERIT)
	if err != nil {
		return syscall.InvalidHandle, os.NewSyscallError("socket", err)
	}
	return s, nil
}

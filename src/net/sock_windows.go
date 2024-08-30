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
	// When the socket backlog is SOMAXCONN, Windows will set the backlog to
	// "a reasonable maximum value".
	// See: https://learn.microsoft.com/en-us/windows/win32/api/winsock2/nf-winsock2-listen
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

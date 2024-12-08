// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements sysSocket for platforms that provide a fast path for
// setting SetNonblock and CloseOnExec, but don't necessarily support it.
// Support for SOCK_* flags as part of the type parameter was added to Oracle
// Solaris in the 11.4 release. Thus, on releases prior to 11.4, we fall back
// to the combination of socket(3c) and fcntl(2).

package net

import (
	"internal/poll"
	"internal/syscall/unix"
	"os"
	"syscall"
)

// Wrapper around the socket system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func sysSocket(family, sotype, proto int) (int, error) {
	// Perform a cheap test and try the fast path first.
	if unix.SupportSockNonblockCloexec() {
		s, err := socketFunc(family, sotype|syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC, proto)
		if err != nil {
			return -1, os.NewSyscallError("socket", err)
		}
		return s, nil
	}

	// See ../syscall/exec_unix.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err := socketFunc(family, sotype, proto)
	if err == nil {
		syscall.CloseOnExec(s)
	}
	syscall.ForkLock.RUnlock()
	if err != nil {
		return -1, os.NewSyscallError("socket", err)
	}
	if err = syscall.SetNonblock(s, true); err != nil {
		poll.CloseFunc(s)
		return -1, os.NewSyscallError("setnonblock", err)
	}
	return s, nil
}

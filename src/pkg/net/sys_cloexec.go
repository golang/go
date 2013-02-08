// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements sysSocket and accept for platforms that do not
// provide a fast path for setting SetNonblock and CloseOnExec.

// +build darwin freebsd netbsd openbsd

package net

import "syscall"

// Wrapper around the socket system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func sysSocket(f, t, p int) (int, error) {
	// See ../syscall/exec_unix.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err := syscall.Socket(f, t, p)
	if err == nil {
		syscall.CloseOnExec(s)
	}
	syscall.ForkLock.RUnlock()
	if err != nil {
		return -1, err
	}
	if err = syscall.SetNonblock(s, true); err != nil {
		syscall.Close(s)
		return -1, err
	}
	return s, nil
}

// Wrapper around the accept system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func accept(fd int) (int, syscall.Sockaddr, error) {
	// See ../syscall/exec_unix.go for description of ForkLock.
	// It is probably okay to hold the lock across syscall.Accept
	// because we have put fd.sysfd into non-blocking mode.
	// However, a call to the File method will put it back into
	// blocking mode. We can't take that risk, so no use of ForkLock here.
	nfd, sa, err := syscall.Accept(fd)
	if err == nil {
		syscall.CloseOnExec(nfd)
	}
	if err != nil {
		return -1, nil, err
	}
	if err = syscall.SetNonblock(nfd, true); err != nil {
		syscall.Close(nfd)
		return -1, nil, err
	}
	return nfd, sa, nil
}

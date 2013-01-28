// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements sysSocket and accept for platforms that
// provide a fast path for setting SetNonblock and CloseOnExec.

// +build linux

package net

import "syscall"

// Wrapper around the socket system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func sysSocket(f, t, p int) (int, error) {
	s, err := syscall.Socket(f, t|syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC, p)
	// The SOCK_NONBLOCK and SOCK_CLOEXEC flags were introduced in
	// Linux 2.6.27.  If we get an EINVAL error, fall back to
	// using socket without them.
	if err == nil || err != syscall.EINVAL {
		return s, err
	}

	// See ../syscall/exec_unix.go for description of ForkLock.
	syscall.ForkLock.RLock()
	s, err = syscall.Socket(f, t, p)
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
	nfd, sa, err := syscall.Accept4(fd, syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC)
	// The accept4 system call was introduced in Linux 2.6.28.  If
	// we get an ENOSYS error, fall back to using accept.
	if err == nil || err != syscall.ENOSYS {
		return nfd, sa, err
	}

	// See ../syscall/exec_unix.go for description of ForkLock.
	// It is okay to hold the lock across syscall.Accept
	// because we have put fd.sysfd into non-blocking mode.
	syscall.ForkLock.RLock()
	nfd, sa, err = syscall.Accept(fd)
	if err == nil {
		syscall.CloseOnExec(nfd)
	}
	syscall.ForkLock.RUnlock()
	if err != nil {
		return -1, nil, err
	}
	if err = syscall.SetNonblock(nfd, true); err != nil {
		syscall.Close(nfd)
		return -1, nil, err
	}
	return nfd, sa, nil
}

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements accept for platforms that provide a fast path for
// setting SetNonblock and CloseOnExec, but don't necessarily have accept4.
// The accept4(3c) function was added to Oracle Solaris in the Solaris 11.4.0
// release. Thus, on releases prior to 11.4, we fall back to the combination
// of accept(3c) and fcntl(2).

package poll

import (
	"internal/syscall/unix"
	"syscall"
)

// Wrapper around the accept system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func accept(s int) (int, syscall.Sockaddr, string, error) {
	// Perform a cheap test and try the fast path first.
	if unix.SupportAccept4() {
		ns, sa, err := Accept4Func(s, syscall.SOCK_NONBLOCK|syscall.SOCK_CLOEXEC)
		if err != nil {
			return -1, nil, "accept4", err
		}
		return ns, sa, "", nil
	}

	// See ../syscall/exec_unix.go for description of ForkLock.
	// It is probably okay to hold the lock across syscall.Accept
	// because we have put fd.sysfd into non-blocking mode.
	// However, a call to the File method will put it back into
	// blocking mode. We can't take that risk, so no use of ForkLock here.
	ns, sa, err := AcceptFunc(s)
	if err == nil {
		syscall.CloseOnExec(ns)
	}
	if err != nil {
		return -1, nil, "accept", err
	}
	if err = syscall.SetNonblock(ns, true); err != nil {
		CloseFunc(ns)
		return -1, nil, "setnonblock", err
	}
	return ns, sa, "", nil
}

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements sysSocket and accept for platforms that do not
// provide a fast path for setting SetNonblock and CloseOnExec.

// +build aix darwin js,wasm solaris,!illumos

package poll

import (
	"syscall"
)

// Wrapper around the accept system call that marks the returned file
// descriptor as nonblocking and close-on-exec.
func accept(s int) (int, syscall.Sockaddr, string, error) {
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

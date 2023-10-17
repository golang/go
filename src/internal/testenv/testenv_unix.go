// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package testenv

import (
	"errors"
	"io/fs"
	"syscall"
)

// Sigquit is the signal to send to kill a hanging subprocess.
// Send SIGQUIT to get a stack trace.
var Sigquit = syscall.SIGQUIT

func syscallIsNotSupported(err error) bool {
	if err == nil {
		return false
	}

	var errno syscall.Errno
	if errors.As(err, &errno) {
		switch errno {
		case syscall.EPERM, syscall.EROFS:
			// User lacks permission: either the call requires root permission and the
			// user is not root, or the call is denied by a container security policy.
			return true
		case syscall.EINVAL:
			// Some containers return EINVAL instead of EPERM if a system call is
			// denied by security policy.
			return true
		}
	}

	if errors.Is(err, fs.ErrPermission) || errors.Is(err, errors.ErrUnsupported) {
		return true
	}

	return false
}

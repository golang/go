// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testpty

import (
	"internal/syscall/unix"
	"os"
	"syscall"
)

func open() (pty *os.File, processTTY string, err error) {
	m, err := unix.PosixOpenpt(syscall.O_RDWR)
	if err != nil {
		return nil, "", ptyError("posix_openpt", err)
	}
	if err := unix.Grantpt(m); err != nil {
		syscall.Close(m)
		return nil, "", ptyError("grantpt", err)
	}
	if err := unix.Unlockpt(m); err != nil {
		syscall.Close(m)
		return nil, "", ptyError("unlockpt", err)
	}
	processTTY, err = unix.Ptsname(m)
	if err != nil {
		syscall.Close(m)
		return nil, "", ptyError("ptsname", err)
	}
	return os.NewFile(uintptr(m), "pty"), processTTY, nil
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package main

import (
	"syscall"
	"unsafe"
)

const ioctlReadTermios = syscall.TCGETS

// isTerminal reports whether fd is a terminal.
func isTerminal(fd uintptr) bool {
	var termios syscall.Termios
	_, _, err := syscall.Syscall6(syscall.SYS_IOCTL, fd, ioctlReadTermios, uintptr(unsafe.Pointer(&termios)), 0, 0, 0)
	return err == 0
}

func init() {
	stdOutErrAreTerminals = func() bool {
		return isTerminal(1) && isTerminal(2)
	}
}

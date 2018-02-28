// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

// This test is run by src/cmd/dist/test.go (cmd_go_test_terminal),
// and not by cmd/go's tests. This is because this test requires that
// that it be called with its stdout and stderr being a terminal.
// dist doesn't run `cmd/go test` against this test directory if
// dist's stdout/stderr aren't terminals.
//
// See issue 18153.

package p

import (
	"syscall"
	"testing"
	"unsafe"
)

const ioctlReadTermios = syscall.TCGETS

// isTerminal reports whether fd is a terminal.
func isTerminal(fd uintptr) bool {
	var termios syscall.Termios
	_, _, err := syscall.Syscall6(syscall.SYS_IOCTL, fd, ioctlReadTermios, uintptr(unsafe.Pointer(&termios)), 0, 0, 0)
	return err == 0
}

func TestIsTerminal(t *testing.T) {
	if !isTerminal(1) {
		t.Errorf("stdout is not a terminal")
	}
	if !isTerminal(2) {
		t.Errorf("stderr is not a terminal")
	}
}

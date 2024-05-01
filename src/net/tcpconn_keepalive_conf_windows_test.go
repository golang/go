// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build windows

package net

import (
	"internal/syscall/windows"
	"syscall"
	"testing"
)

const (
	syscall_TCP_KEEPIDLE  = windows.TCP_KEEPIDLE
	syscall_TCP_KEEPCNT   = windows.TCP_KEEPCNT
	syscall_TCP_KEEPINTVL = windows.TCP_KEEPINTVL
)

type fdType = syscall.Handle

func maybeSkipKeepAliveTest(t *testing.T) {
	// TODO(panjf2000): Unlike Unix-like OS's, old Windows (prior to Windows 10, version 1709)
	// 	doesn't provide any ways to retrieve the current TCP keep-alive settings, therefore
	// 	we're not able to run the test suite similar to Unix-like OS's on Windows.
	//  Try to find another proper approach to test the keep-alive settings on old Windows.
	if !windows.SupportTCPKeepAliveIdle() || !windows.SupportTCPKeepAliveInterval() || !windows.SupportTCPKeepAliveCount() {
		t.Skip("skipping on windows")
	}
}

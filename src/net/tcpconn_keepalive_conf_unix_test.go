// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || dragonfly || freebsd || linux || netbsd || solaris

package net

import (
	"runtime"
	"syscall"
	"testing"
)

const (
	syscall_TCP_KEEPIDLE  = syscall.TCP_KEEPIDLE
	syscall_TCP_KEEPCNT   = syscall.TCP_KEEPCNT
	syscall_TCP_KEEPINTVL = syscall.TCP_KEEPINTVL
)

type fdType = int

func maybeSkipKeepAliveTest(t *testing.T) {
	// TODO(panjf2000): stop skipping this test on Solaris
	//  when https://go.dev/issue/64251 is fixed.
	if runtime.GOOS == "solaris" {
		t.Skip("skipping on solaris for now")
	}
}

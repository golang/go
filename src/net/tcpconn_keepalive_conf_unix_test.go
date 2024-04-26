// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build aix || dragonfly || freebsd || illumos || linux || netbsd

package net

import (
	"syscall"
	"testing"
)

const (
	syscall_TCP_KEEPIDLE  = syscall.TCP_KEEPIDLE
	syscall_TCP_KEEPCNT   = syscall.TCP_KEEPCNT
	syscall_TCP_KEEPINTVL = syscall.TCP_KEEPINTVL
)

type fdType = int

func maybeSkipKeepAliveTest(_ *testing.T) {}

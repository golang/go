// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin

package net

import (
	"syscall"
	"testing"
)

const (
	syscall_TCP_KEEPIDLE  = syscall.TCP_KEEPALIVE
	syscall_TCP_KEEPCNT   = sysTCP_KEEPCNT
	syscall_TCP_KEEPINTVL = sysTCP_KEEPINTVL
)

type fdType = int

func maybeSkipKeepAliveTest(_ *testing.T) {}

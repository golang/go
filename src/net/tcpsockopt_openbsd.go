// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"
	"time"
)

func setKeepAliveIdle(_ *netFD, d time.Duration) error {
	if d < 0 {
		return nil
	}
	// OpenBSD has no user-settable per-socket TCP keepalive
	// options.
	return syscall.ENOPROTOOPT
}

func setKeepAliveInterval(_ *netFD, d time.Duration) error {
	if d < 0 {
		return nil
	}
	// OpenBSD has no user-settable per-socket TCP keepalive
	// options.
	return syscall.ENOPROTOOPT
}

func setKeepAliveCount(_ *netFD, n int) error {
	if n < 0 {
		return nil
	}
	// OpenBSD has no user-settable per-socket TCP keepalive
	// options.
	return syscall.ENOPROTOOPT
}

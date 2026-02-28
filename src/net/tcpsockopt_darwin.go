// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"syscall"
	"time"
)

// syscall.TCP_KEEPINTVL and syscall.TCP_KEEPCNT might be missing on some darwin architectures.
const (
	sysTCP_KEEPINTVL = 0x101
	sysTCP_KEEPCNT   = 0x102
)

func setKeepAliveIdle(fd *netFD, d time.Duration) error {
	if d == 0 {
		d = defaultTCPKeepAliveIdle
	} else if d < 0 {
		return nil
	}

	// The kernel expects seconds so round to next highest second.
	secs := int(roundDurationUp(d, time.Second))
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE, secs)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setKeepAliveInterval(fd *netFD, d time.Duration) error {
	if d == 0 {
		d = defaultTCPKeepAliveInterval
	} else if d < 0 {
		return nil
	}

	// The kernel expects seconds so round to next highest second.
	secs := int(roundDurationUp(d, time.Second))
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, sysTCP_KEEPINTVL, secs)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setKeepAliveCount(fd *netFD, n int) error {
	if n == 0 {
		n = defaultTCPKeepAliveCount
	} else if n < 0 {
		return nil
	}

	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, sysTCP_KEEPCNT, n)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

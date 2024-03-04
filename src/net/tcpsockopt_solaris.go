// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"syscall"
	"time"
)

func setKeepAliveIdle(fd *netFD, d time.Duration) error {
	if d == 0 {
		d = defaultTCPKeepAliveIdle
	} else if d < 0 {
		return nil
	}

	// The kernel expects milliseconds so round to next highest
	// millisecond.
	msecs := int(roundDurationUp(d, time.Millisecond))

	// TODO(panjf2000): the system call here always returns an error of invalid argument,
	//	 this was never discovered due to the lack of tests for TCP keep-alive on various
	//	 platforms in Go's test suite. Try to dive deep and figure out the reason later.
	// Check out https://go.dev/issue/64251 for more details.
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_THRESHOLD, msecs)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setKeepAliveInterval(_ *netFD, d time.Duration) error {
	if d < 0 {
		return nil
	}

	// Normally we'd do
	//	syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_TCP, syscall.TCP_KEEPINTVL, secs)
	// here, but we can't because Solaris does not have TCP_KEEPINTVL.
	// Solaris has TCP_KEEPALIVE_ABORT_THRESHOLD, but it's not the same
	// thing, it refers to the total time until aborting (not between
	// probes), and it uses an exponential backoff algorithm instead of
	// waiting the same time between probes. We can't hope for the best
	// and do it anyway, like on Darwin, because Solaris might eventually
	// allocate a constant with a different meaning for the value of
	// TCP_KEEPINTVL on illumos.
	return syscall.ENOPROTOOPT
}

func setKeepAliveCount(_ *netFD, n int) error {
	if n < 0 {
		return nil
	}
	return syscall.ENOPROTOOPT
}

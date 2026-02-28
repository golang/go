// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !illumos

package net

import (
	"internal/syscall/unix"
	"runtime"
	"syscall"
	"time"
)

// Some macros of TCP Keep-Alive options on Solaris 11.4 may
// differ from those on OpenSolaris-based derivatives.
const (
	sysTCP_KEEPIDLE  = 0x1D
	sysTCP_KEEPINTVL = 0x1E
	sysTCP_KEEPCNT   = 0x1F
)

func setKeepAliveIdle(fd *netFD, d time.Duration) error {
	if !unix.SupportTCPKeepAliveIdleIntvlCNT() {
		return setKeepAliveIdleAndIntervalAndCount(fd, d, -1, -1)
	}

	if d == 0 {
		d = defaultTCPKeepAliveIdle
	} else if d < 0 {
		return nil
	}
	// The kernel expects seconds so round to next highest second.
	secs := int(roundDurationUp(d, time.Second))
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, sysTCP_KEEPIDLE, secs)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setKeepAliveInterval(fd *netFD, d time.Duration) error {
	if !unix.SupportTCPKeepAliveIdleIntvlCNT() {
		return syscall.EPROTOTYPE
	}

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
	if !unix.SupportTCPKeepAliveIdleIntvlCNT() {
		return syscall.EPROTOTYPE
	}

	if n == 0 {
		n = defaultTCPKeepAliveCount
	} else if n < 0 {
		return nil
	}
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, sysTCP_KEEPCNT, n)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

// setKeepAliveIdleAndIntervalAndCount serves for Solaris prior to 11.4 by simulating
// the TCP_KEEPIDLE, TCP_KEEPINTVL, and TCP_KEEPCNT with `TCP_KEEPALIVE_THRESHOLD` + `TCP_KEEPALIVE_ABORT_THRESHOLD`.
func setKeepAliveIdleAndIntervalAndCount(fd *netFD, idle, interval time.Duration, count int) error {
	if idle == 0 {
		idle = defaultTCPKeepAliveIdle
	}

	// The kernel expects milliseconds so round to next highest
	// millisecond.
	if idle > 0 {
		msecs := int(roundDurationUp(idle, time.Millisecond))
		err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_THRESHOLD, msecs)
		runtime.KeepAlive(fd)
		if err != nil {
			return wrapSyscallError("setsockopt", err)
		}
	}

	if interval == 0 {
		interval = defaultTCPKeepAliveInterval
	}
	if count == 0 {
		count = defaultTCPKeepAliveCount
	}
	// TCP_KEEPINTVL and TCP_KEEPCNT are not available on Solaris
	// prior to 11.4, so it's pointless to "leave it unchanged"
	// with negative value for only one of them. On the other hand,
	// setting both to negative values should pragmatically leave the
	// TCP_KEEPALIVE_ABORT_THRESHOLD unchanged.
	abortIdle := int(roundDurationUp(interval, time.Millisecond)) * count
	if abortIdle < 0 {
		return syscall.ENOPROTOOPT
	}
	if interval < 0 && count < 0 {
		abortIdle = -1
	}

	if abortIdle > 0 {
		// Note that the consequent probes will not be sent at equal intervals on Solaris,
		// but will be sent using the exponential backoff algorithm.
		err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE_ABORT_THRESHOLD, abortIdle)
		runtime.KeepAlive(fd)
		return wrapSyscallError("setsockopt", err)
	}

	return nil
}

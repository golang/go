// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"syscall"
	"time"
)

func setKeepAlivePeriod(fd *netFD, d time.Duration) error {
	// The kernel expects milliseconds so round to next highest
	// millisecond.
	msecs := int(roundDurationUp(d, time.Millisecond))
	if err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, syscall.TCP_KEEPINTVL, msecs); err != nil {
		return wrapSyscallError("setsockopt", err)
	}
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_TCP, syscall.TCP_KEEPIDLE, msecs)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

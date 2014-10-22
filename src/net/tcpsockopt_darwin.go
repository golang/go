// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"syscall"
	"time"
)

const sysTCP_KEEPINTVL = 0x101

func setKeepAlivePeriod(fd *netFD, d time.Duration) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	// The kernel expects seconds so round to next highest second.
	d += (time.Second - time.Nanosecond)
	secs := int(d.Seconds())
	switch err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_TCP, sysTCP_KEEPINTVL, secs); err {
	case nil, syscall.ENOPROTOOPT: // OS X 10.7 and earlier don't support this option
	default:
		return os.NewSyscallError("setsockopt", err)
	}
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_TCP, syscall.TCP_KEEPALIVE, secs))
}

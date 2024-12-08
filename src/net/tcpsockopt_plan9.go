// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TCP socket options for plan9

package net

import (
	"internal/itoa"
	"syscall"
	"time"
)

func setNoDelay(_ *netFD, _ bool) error {
	return syscall.EPLAN9
}

// Set keep alive period.
func setKeepAliveIdle(fd *netFD, d time.Duration) error {
	if d < 0 {
		return nil
	}

	cmd := "keepalive " + itoa.Itoa(int(d/time.Millisecond))
	_, e := fd.ctl.WriteAt([]byte(cmd), 0)
	return e
}

func setKeepAliveInterval(_ *netFD, d time.Duration) error {
	if d < 0 {
		return nil
	}
	return syscall.EPLAN9
}

func setKeepAliveCount(_ *netFD, n int) error {
	if n < 0 {
		return nil
	}
	return syscall.EPLAN9
}

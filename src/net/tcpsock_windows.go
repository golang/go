// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/syscall/windows"
	"syscall"
)

// SetKeepAliveConfig configures keep-alive messages sent by the operating system.
func (c *TCPConn) SetKeepAliveConfig(config KeepAliveConfig) error {
	if !c.ok() {
		return syscall.EINVAL
	}

	if err := setKeepAlive(c.fd, config.Enable); err != nil {
		return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	if windows.SupportTCPKeepAliveIdle() && windows.SupportTCPKeepAliveInterval() {
		if err := setKeepAliveIdle(c.fd, config.Idle); err != nil {
			return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
		}
		if err := setKeepAliveInterval(c.fd, config.Interval); err != nil {
			return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
		}
	} else if err := setKeepAliveIdleAndInterval(c.fd, config.Idle, config.Interval); err != nil {
		return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	if err := setKeepAliveCount(c.fd, config.Count); err != nil {
		return &OpError{Op: "set", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}

	return nil
}

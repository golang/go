// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

// Default values of KeepAliveTime and KeepAliveInterval on Windows,
// check out https://learn.microsoft.com/en-us/windows/win32/winsock/sio-keepalive-vals#remarks for details.
const (
	defaultKeepAliveIdle     = 2 * time.Hour
	defaultKeepAliveInterval = time.Second
)

func setKeepAliveIdle(fd *netFD, d time.Duration) error {
	return setKeepAliveIdleAndInterval(fd, d, -1)
}

func setKeepAliveInterval(fd *netFD, d time.Duration) error {
	return setKeepAliveIdleAndInterval(fd, -1, d)
}

func setKeepAliveCount(_ *netFD, n int) error {
	if n < 0 {
		return nil
	}

	// This value is not capable to be changed on Windows.
	return syscall.WSAENOPROTOOPT
}

func setKeepAliveIdleAndInterval(fd *netFD, idle, interval time.Duration) error {
	// WSAIoctl with SIO_KEEPALIVE_VALS control code requires all fields in
	// `tcp_keepalive` struct to be provided.
	// Otherwise, if any of the fields were not provided, just leaving them
	// zero will knock off any existing values of keep-alive.
	// Unfortunately, Windows doesn't support retrieving current keep-alive
	// settings in any form programmatically, which disable us to first retrieve
	// the current keep-alive settings, then set it without unwanted corruption.
	switch {
	case idle < 0 && interval >= 0:
		// Given that we can't set KeepAliveInterval alone, and this code path
		// is new, it doesn't exist before, so we just return an error.
		return syscall.WSAENOPROTOOPT
	case idle >= 0 && interval < 0:
		// Although we can't set KeepAliveTime alone either, this existing code
		// path had been backing up [SetKeepAlivePeriod] which used to be set both
		// KeepAliveTime and KeepAliveInterval to 15 seconds.
		// Now we will use the default of KeepAliveInterval on Windows if user doesn't
		// provide one.
		interval = defaultKeepAliveInterval
	case idle < 0 && interval < 0:
		// Nothing to do, just bail out.
		return nil
	case idle >= 0 && interval >= 0:
		// Go ahead.
	}

	if idle == 0 {
		idle = defaultTCPKeepAliveIdle
	}
	if interval == 0 {
		interval = defaultTCPKeepAliveInterval
	}

	// The kernel expects milliseconds so round to next highest
	// millisecond.
	tcpKeepAliveIdle := uint32(roundDurationUp(idle, time.Millisecond))
	tcpKeepAliveInterval := uint32(roundDurationUp(interval, time.Millisecond))
	ka := syscall.TCPKeepalive{
		OnOff:    1,
		Time:     tcpKeepAliveIdle,
		Interval: tcpKeepAliveInterval,
	}
	ret := uint32(0)
	size := uint32(unsafe.Sizeof(ka))
	err := fd.pfd.WSAIoctl(syscall.SIO_KEEPALIVE_VALS, (*byte)(unsafe.Pointer(&ka)), size, nil, 0, &ret, nil, 0)
	runtime.KeepAlive(fd)
	return os.NewSyscallError("wsaioctl", err)
}

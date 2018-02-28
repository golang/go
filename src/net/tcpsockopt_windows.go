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

func setKeepAlivePeriod(fd *netFD, d time.Duration) error {
	// The kernel expects milliseconds so round to next highest
	// millisecond.
	d += (time.Millisecond - time.Nanosecond)
	msecs := uint32(d / time.Millisecond)
	ka := syscall.TCPKeepalive{
		OnOff:    1,
		Time:     msecs,
		Interval: msecs,
	}
	ret := uint32(0)
	size := uint32(unsafe.Sizeof(ka))
	err := fd.pfd.WSAIoctl(syscall.SIO_KEEPALIVE_VALS, (*byte)(unsafe.Pointer(&ka)), size, nil, 0, &ret, nil, 0)
	runtime.KeepAlive(fd)
	return os.NewSyscallError("wsaioctl", err)
}

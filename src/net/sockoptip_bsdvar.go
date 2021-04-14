// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd netbsd openbsd solaris

package net

import (
	"runtime"
	"syscall"
)

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	ip, err := interfaceToIPv4Addr(ifi)
	if err != nil {
		return wrapSyscallError("setsockopt", err)
	}
	var a [4]byte
	copy(a[:], ip.To4())
	err = fd.pfd.SetsockoptInet4Addr(syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, a)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	err := fd.pfd.SetsockoptByte(syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, byte(boolint(v)))
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

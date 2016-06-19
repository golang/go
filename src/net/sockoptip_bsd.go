// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package net

import (
	"os"
	"syscall"
)

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	ip, err := interfaceToIPv4Addr(ifi)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	var a [4]byte
	copy(a[:], ip.To4())
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInet4Addr(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, a))
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptByte(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, byte(boolint(v))))
}

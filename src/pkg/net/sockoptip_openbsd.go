// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP-level socket options for OpenBSD

package net

import (
	"os"
	"syscall"
)

func ipv4MulticastInterface(fd *netFD) (*Interface, error) {
	fd.incref()
	defer fd.decref()
	a, err := syscall.GetsockoptInet4Addr(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF)
	if err != nil {
		return nil, os.NewSyscallError("getsockopt", err)
	}
	return ipv4AddrToInterface(IPv4(a[0], a[1], a[2], a[3]))
}

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	ip, err := interfaceToIPv4Addr(ifi)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	var x [4]byte
	copy(x[:], ip.To4())
	fd.incref()
	defer fd.decref()
	err = syscall.SetsockoptInet4Addr(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, x)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv4MulticastLoopback(fd *netFD) (bool, error) {
	fd.incref()
	defer fd.decref()
	v, err := syscall.GetsockoptByte(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP)
	if err != nil {
		return false, os.NewSyscallError("getsockopt", err)
	}
	return v == 1, nil
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	fd.incref()
	defer fd.decref()
	err := syscall.SetsockoptByte(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, byte(boolint(v)))
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv4ReceiveInterface(fd *netFD) (bool, error) {
	fd.incref()
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_RECVIF)
	if err != nil {
		return false, os.NewSyscallError("getsockopt", err)
	}
	return v == 1, nil
}

func setIPv4ReceiveInterface(fd *netFD, v bool) error {
	fd.incref()
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_RECVIF, boolint(v))
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

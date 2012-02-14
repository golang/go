// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd netbsd openbsd

// IP-level socket options for BSD variants

package net

import (
	"os"
	"syscall"
)

func ipv4MulticastTTL(fd *netFD) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptByte(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_TTL)
	if err != nil {
		return 0, os.NewSyscallError("getsockopt", err)
	}
	return int(v), nil
}

func setIPv4MulticastTTL(fd *netFD, v int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptByte(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_TTL, byte(v))
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv6TrafficClass(fd *netFD) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_TCLASS)
	if err != nil {
		return 0, os.NewSyscallError("getsockopt", err)
	}
	return v, nil
}

func setIPv6TrafficClass(fd *netFD, v int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_TCLASS, v)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

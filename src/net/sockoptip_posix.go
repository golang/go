// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd windows

package net

import (
	"os"
	"syscall"
)

func joinIPv4Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPMreq{Multiaddr: [4]byte{ip[0], ip[1], ip[2], ip[3]}}
	if err := setIPv4MreqToInterface(mreq, ifi); err != nil {
		return err
	}
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptIPMreq(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_ADD_MEMBERSHIP, mreq))
}

func setIPv6MulticastInterface(fd *netFD, ifi *Interface) error {
	var v int
	if ifi != nil {
		v = ifi.Index
	}
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_IF, v))
}

func setIPv6MulticastLoopback(fd *netFD, v bool) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_LOOP, boolint(v)))
}

func joinIPv6Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPv6Mreq{}
	copy(mreq.Multiaddr[:], ip)
	if ifi != nil {
		mreq.Interface = uint32(ifi.Index)
	}
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptIPv6Mreq(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_JOIN_GROUP, mreq))
}

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// IP-level socket options

package net

import (
	"os"
	"syscall"
)

func ipv4TOS(fd *netFD) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_TOS)
	if err != nil {
		return 0, os.NewSyscallError("getsockopt", err)
	}
	return v, nil
}

func setIPv4TOS(fd *netFD, v int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_TOS, v)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv4TTL(fd *netFD) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_TTL)
	if err != nil {
		return 0, os.NewSyscallError("getsockopt", err)
	}
	return v, nil
}

func setIPv4TTL(fd *netFD, v int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_TTL, v)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func joinIPv4Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPMreq{Multiaddr: [4]byte{ip[0], ip[1], ip[2], ip[3]}}
	if err := setIPv4MreqToInterface(mreq, ifi); err != nil {
		return err
	}
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptIPMreq(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_ADD_MEMBERSHIP, mreq))
}

func leaveIPv4Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPMreq{Multiaddr: [4]byte{ip[0], ip[1], ip[2], ip[3]}}
	if err := setIPv4MreqToInterface(mreq, ifi); err != nil {
		return err
	}
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptIPMreq(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_DROP_MEMBERSHIP, mreq))
}

func ipv6HopLimit(fd *netFD) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_UNICAST_HOPS)
	if err != nil {
		return 0, os.NewSyscallError("getsockopt", err)
	}
	return v, nil
}

func setIPv6HopLimit(fd *netFD, v int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_UNICAST_HOPS, v)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv6MulticastInterface(fd *netFD) (*Interface, error) {
	if err := fd.incref(false); err != nil {
		return nil, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_IF)
	if err != nil {
		return nil, os.NewSyscallError("getsockopt", err)
	}
	if v == 0 {
		return nil, nil
	}
	ifi, err := InterfaceByIndex(v)
	if err != nil {
		return nil, err
	}
	return ifi, nil
}

func setIPv6MulticastInterface(fd *netFD, ifi *Interface) error {
	var v int
	if ifi != nil {
		v = ifi.Index
	}
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_IF, v)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv6MulticastHopLimit(fd *netFD) (int, error) {
	if err := fd.incref(false); err != nil {
		return 0, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_HOPS)
	if err != nil {
		return 0, os.NewSyscallError("getsockopt", err)
	}
	return v, nil
}

func setIPv6MulticastHopLimit(fd *netFD, v int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_HOPS, v)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv6MulticastLoopback(fd *netFD) (bool, error) {
	if err := fd.incref(false); err != nil {
		return false, err
	}
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_LOOP)
	if err != nil {
		return false, os.NewSyscallError("getsockopt", err)
	}
	return v == 1, nil
}

func setIPv6MulticastLoopback(fd *netFD, v bool) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_LOOP, boolint(v))
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func joinIPv6Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPv6Mreq{}
	copy(mreq.Multiaddr[:], ip)
	if ifi != nil {
		mreq.Interface = uint32(ifi.Index)
	}
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptIPv6Mreq(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_JOIN_GROUP, mreq))
}

func leaveIPv6Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPv6Mreq{}
	copy(mreq.Multiaddr[:], ip)
	if ifi != nil {
		mreq.Interface = uint32(ifi.Index)
	}
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptIPv6Mreq(fd.sysfd, syscall.IPPROTO_IPV6, syscall.IPV6_LEAVE_GROUP, mreq))
}

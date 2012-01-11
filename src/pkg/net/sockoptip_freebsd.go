// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP-level socket options for FreeBSD

package net

import (
	"os"
	"syscall"
)

func ipv4MulticastInterface(fd *netFD) (*Interface, error) {
	fd.incref()
	defer fd.decref()
	mreq, err := syscall.GetsockoptIPMreqn(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF)
	if err != nil {
		return nil, os.NewSyscallError("getsockopt", err)
	}
	if int(mreq.Ifindex) == 0 {
		return nil, nil
	}
	return InterfaceByIndex(int(mreq.Ifindex))
}

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	var v int32
	if ifi != nil {
		v = int32(ifi.Index)
	}
	mreq := &syscall.IPMreqn{Ifindex: v}
	fd.incref()
	defer fd.decref()
	err := syscall.SetsockoptIPMreqn(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, mreq)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv4MulticastLoopback(fd *netFD) (bool, error) {
	fd.incref()
	defer fd.decref()
	v, err := syscall.GetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP)
	if err != nil {
		return false, os.NewSyscallError("getsockopt", err)
	}
	return v == 1, nil
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	fd.incref()
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, boolint(v))
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

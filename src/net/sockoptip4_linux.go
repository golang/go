// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"runtime"
	"syscall"
)

func joinIPv4Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPMreqn{Multiaddr: [4]byte{ip[0], ip[1], ip[2], ip[3]}}
	if ifi != nil {
		mreq.Ifindex = int32(ifi.Index)
	}
	err := fd.pfd.SetsockoptIPMreqn(syscall.IPPROTO_IP, syscall.IP_ADD_MEMBERSHIP, mreq)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	var v int32
	if ifi != nil {
		v = int32(ifi.Index)
	}
	mreq := &syscall.IPMreqn{Ifindex: v}
	err := fd.pfd.SetsockoptIPMreqn(syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, mreq)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, boolint(v))
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

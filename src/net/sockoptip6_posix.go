// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || windows

package net

import (
	"runtime"
	"syscall"
)

func setIPv6MulticastInterface(fd *netFD, ifi *Interface) error {
	var v int
	if ifi != nil {
		v = ifi.Index
	}
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_IF, v)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func setIPv6MulticastLoopback(fd *netFD, v bool) error {
	err := fd.pfd.SetsockoptInt(syscall.IPPROTO_IPV6, syscall.IPV6_MULTICAST_LOOP, boolint(v))
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

func joinIPv6Group(fd *netFD, ifi *Interface, ip IP) error {
	mreq := &syscall.IPv6Mreq{}
	copy(mreq.Multiaddr[:], ip)
	if ifi != nil {
		mreq.Interface = uint32(ifi.Index)
	}
	err := fd.pfd.SetsockoptIPv6Mreq(syscall.IPPROTO_IPV6, syscall.IPV6_JOIN_GROUP, mreq)
	runtime.KeepAlive(fd)
	return wrapSyscallError("setsockopt", err)
}

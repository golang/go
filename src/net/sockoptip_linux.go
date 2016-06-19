// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"syscall"
)

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	var v int32
	if ifi != nil {
		v = int32(ifi.Index)
	}
	mreq := &syscall.IPMreqn{Ifindex: v}
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptIPMreqn(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, mreq))
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, boolint(v)))
}

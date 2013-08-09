// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"syscall"
	"unsafe"
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
	return os.NewSyscallError("setsockopt", syscall.Setsockopt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, (*byte)(unsafe.Pointer(&a[0])), 4))
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	if err := fd.incref(); err != nil {
		return err
	}
	defer fd.decref()
	return os.NewSyscallError("setsockopt", syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, boolint(v)))
}

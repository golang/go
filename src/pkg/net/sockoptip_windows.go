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
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err = syscall.Setsockopt(fd.sysfd, int32(syscall.IPPROTO_IP), int32(syscall.IP_MULTICAST_IF), (*byte)(unsafe.Pointer(&a[0])), 4)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	vv := int32(boolint(v))
	err := syscall.Setsockopt(fd.sysfd, int32(syscall.IPPROTO_IP), int32(syscall.IP_MULTICAST_LOOP), (*byte)(unsafe.Pointer(&vv)), 4)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP-level socket options for Windows

package net

import (
	"os"
	"syscall"
)

func ipv4MulticastInterface(fd *netFD) (*Interface, error) {
	// TODO: Implement this
	return nil, syscall.EWINDOWS
}

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	ip, err := interfaceToIPv4Addr(ifi)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	var x [4]byte
	copy(x[:], ip.To4())
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err = syscall.SetsockoptInet4Addr(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_IF, x)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil
}

func ipv4MulticastTTL(fd *netFD) (int, error) {
	// TODO: Implement this
	return -1, syscall.EWINDOWS
}

func setIPv4MulticastTTL(fd *netFD, v int) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_TTL, v)
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil

}

func ipv4MulticastLoopback(fd *netFD) (bool, error) {
	// TODO: Implement this
	return false, syscall.EWINDOWS
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	if err := fd.incref(false); err != nil {
		return err
	}
	defer fd.decref()
	err := syscall.SetsockoptInt(fd.sysfd, syscall.IPPROTO_IP, syscall.IP_MULTICAST_LOOP, boolint(v))
	if err != nil {
		return os.NewSyscallError("setsockopt", err)
	}
	return nil

}

func ipv4ReceiveInterface(fd *netFD) (bool, error) {
	// TODO: Implement this
	return false, syscall.EWINDOWS
}

func setIPv4ReceiveInterface(fd *netFD, v bool) error {
	// TODO: Implement this
	return syscall.EWINDOWS
}

func ipv6TrafficClass(fd *netFD) (int, error) {
	// TODO: Implement this
	return 0, syscall.EWINDOWS
}

func setIPv6TrafficClass(fd *netFD, v int) error {
	// TODO: Implement this
	return syscall.EWINDOWS
}

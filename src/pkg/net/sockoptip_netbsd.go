// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP-level socket options for NetBSD

package net

import "syscall"

func ipv4MulticastInterface(fd *netFD) (*Interface, error) {
	// TODO: Implement this
	return nil, syscall.EAFNOSUPPORT
}

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	// TODO: Implement this
	return syscall.EAFNOSUPPORT
}

func ipv4MulticastLoopback(fd *netFD) (bool, error) {
	// TODO: Implement this
	return false, syscall.EAFNOSUPPORT
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	// TODO: Implement this
	return syscall.EAFNOSUPPORT
}

func ipv4ReceiveInterface(fd *netFD) (bool, error) {
	// TODO: Implement this
	return false, syscall.EAFNOSUPPORT
}

func setIPv4ReceiveInterface(fd *netFD, v bool) error {
	// TODO: Implement this
	return syscall.EAFNOSUPPORT
}

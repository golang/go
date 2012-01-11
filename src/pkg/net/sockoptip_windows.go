// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP-level socket options for Windows

package net

import (
	"syscall"
)

func ipv4MulticastInterface(fd *netFD) (*Interface, error) {
	// TODO: Implement this
	return nil, syscall.EWINDOWS
}

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	// TODO: Implement this
	return syscall.EWINDOWS
}

func ipv4MulticastTTL(fd *netFD) (int, error) {
	// TODO: Implement this
	return -1, syscall.EWINDOWS
}

func setIPv4MulticastTTL(fd *netFD, v int) error {
	// TODO: Implement this
	return syscall.EWINDOWS
}

func ipv4MulticastLoopback(fd *netFD) (bool, error) {
	// TODO: Implement this
	return false, syscall.EWINDOWS
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	// TODO: Implement this
	return syscall.EWINDOWS
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

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP-level socket options for Windows

package net

import (
	"os"
)

func ipv4MulticastInterface(fd *netFD) (*Interface, error) {
	// TODO: Implement this
	return nil, os.EWINDOWS
}

func setIPv4MulticastInterface(fd *netFD, ifi *Interface) error {
	// TODO: Implement this
	return os.EWINDOWS
}

func ipv4MulticastTTL(fd *netFD) (int, error) {
	// TODO: Implement this
	return -1, os.EWINDOWS
}

func setIPv4MulticastTTL(fd *netFD, v int) error {
	// TODO: Implement this
	return os.EWINDOWS
}

func ipv4MultiastLoopback(fd *netFD) (bool, error) {
	// TODO: Implement this
	return false, os.EWINDOWS
}

func setIPv4MulticastLoopback(fd *netFD, v bool) error {
	// TODO: Implement this
	return os.EWINDOWS
}

func ipv4ReceiveInterface(fd *netFD) (bool, error) {
	// TODO: Implement this
	return false, os.EWINDOWS
}

func setIPv4ReceiveInterface(fd *netFD, v bool) error {
	// TODO: Implement this
	return os.EWINDOWS
}

func ipv6TrafficClass(fd *netFD) (int, error) {
	// TODO: Implement this
	return os.EWINDOWS
}

func setIPv6TrafficClass(fd *netFD, v int) error {
	// TODO: Implement this
	return os.EWINDOWS
}

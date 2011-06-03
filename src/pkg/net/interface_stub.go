// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification

package net

import "os"

// IsUp returns true if ifi is up.
func (ifi *Interface) IsUp() bool {
	return false
}

// IsLoopback returns true if ifi is a loopback interface.
func (ifi *Interface) IsLoopback() bool {
	return false
}

// CanBroadcast returns true if ifi supports a broadcast access
// capability.
func (ifi *Interface) CanBroadcast() bool {
	return false
}

// IsPointToPoint returns true if ifi belongs to a point-to-point
// link.
func (ifi *Interface) IsPointToPoint() bool {
	return false
}

// CanMulticast returns true if ifi supports a multicast access
// capability.
func (ifi *Interface) CanMulticast() bool {
	return false
}

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces.  Otheriwse it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, os.Error) {
	return nil, nil
}

// If the ifindex is zero, interfaceAddrTable returns addresses
// for all network interfaces.  Otherwise it returns addresses
// for a specific interface.
func interfaceAddrTable(ifindex int) ([]Addr, os.Error) {
	return nil, nil
}

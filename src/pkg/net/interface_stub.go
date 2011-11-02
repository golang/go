// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build plan9

// Network interface identification

package net

// If the ifindex is zero, interfaceTable returns mappings of all
// network interfaces.  Otheriwse it returns a mapping of a specific
// interface.
func interfaceTable(ifindex int) ([]Interface, error) {
	return nil, nil
}

// If the ifindex is zero, interfaceAddrTable returns addresses
// for all network interfaces.  Otherwise it returns addresses
// for a specific interface.
func interfaceAddrTable(ifindex int) ([]Addr, error) {
	return nil, nil
}

// If the ifindex is zero, interfaceMulticastAddrTable returns
// addresses for all network interfaces.  Otherwise it returns
// addresses for a specific interface.
func interfaceMulticastAddrTable(ifindex int) ([]Addr, error) {
	return nil, nil
}

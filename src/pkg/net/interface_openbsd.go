// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification for OpenBSD

package net

// If the ifindex is zero, interfaceMulticastAddrTable returns
// addresses for all network interfaces.  Otherwise it returns
// addresses for a specific interface.
func interfaceMulticastAddrTable(ifindex int) ([]Addr, error) {
	return nil, nil
}

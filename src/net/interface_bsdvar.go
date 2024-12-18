// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build dragonfly || netbsd || openbsd

package net

import (
	"internal/routebsd"
	"syscall"
)

func interfaceMessages(ifindex int) ([]routebsd.Message, error) {
	rib, err := routebsd.FetchRIB(syscall.AF_UNSPEC, syscall.NET_RT_IFLIST, ifindex)
	if err != nil {
		return nil, err
	}
	return routebsd.ParseRIB(syscall.NET_RT_IFLIST, rib)
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	// TODO(mikio): Implement this like other platforms.
	return nil, nil
}

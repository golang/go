// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Network interface identification for Darwin

package net

import (
	"os"
	"syscall"
)

// If the ifindex is zero, interfaceMulticastAddrTable returns
// addresses for all network interfaces.  Otherwise it returns
// addresses for a specific interface.
func interfaceMulticastAddrTable(ifindex int) ([]Addr, os.Error) {
	var (
		tab   []byte
		e     int
		msgs  []syscall.RoutingMessage
		ifmat []Addr
	)

	tab, e = syscall.RouteRIB(syscall.NET_RT_IFLIST2, ifindex)
	if e != 0 {
		return nil, os.NewSyscallError("route rib", e)
	}

	msgs, e = syscall.ParseRoutingMessage(tab)
	if e != 0 {
		return nil, os.NewSyscallError("route message", e)
	}

	for _, m := range msgs {
		switch v := m.(type) {
		case *syscall.InterfaceMulticastAddrMessage:
			if ifindex == 0 || ifindex == int(v.Header.Index) {
				ifma, err := newMulticastAddr(v)
				if err != nil {
					return nil, err
				}
				ifmat = append(ifmat, ifma...)
			}
		}
	}

	return ifmat, nil
}

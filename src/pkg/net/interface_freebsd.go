// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"syscall"
)

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	tab, err := syscall.RouteRIB(syscall.NET_RT_IFMALIST, ifi.Index)
	if err != nil {
		return nil, os.NewSyscallError("route rib", err)
	}
	msgs, err := syscall.ParseRoutingMessage(tab)
	if err != nil {
		return nil, os.NewSyscallError("route message", err)
	}
	var ifmat []Addr
	for _, m := range msgs {
		switch m := m.(type) {
		case *syscall.InterfaceMulticastAddrMessage:
			if ifi.Index == int(m.Header.Index) {
				ifma, err := newMulticastAddr(ifi, m)
				if err != nil {
					return nil, err
				}
				ifmat = append(ifmat, ifma...)
			}
		}
	}
	return ifmat, nil
}

func newMulticastAddr(ifi *Interface, m *syscall.InterfaceMulticastAddrMessage) ([]Addr, error) {
	sas, err := syscall.ParseRoutingSockaddr(m)
	if err != nil {
		return nil, os.NewSyscallError("route sockaddr", err)
	}
	var ifmat []Addr
	for _, sa := range sas {
		switch sa := sa.(type) {
		case *syscall.SockaddrInet4:
			ifma := &IPAddr{IP: IPv4(sa.Addr[0], sa.Addr[1], sa.Addr[2], sa.Addr[3])}
			ifmat = append(ifmat, ifma.toAddr())
		case *syscall.SockaddrInet6:
			ifma := &IPAddr{IP: make(IP, IPv6len)}
			copy(ifma.IP, sa.Addr[:])
			// NOTE: KAME based IPv6 protocol stack usually embeds
			// the interface index in the interface-local or link-
			// local address as the kernel-internal form.
			if ifma.IP.IsInterfaceLocalMulticast() || ifma.IP.IsLinkLocalMulticast() {
				ifma.IP[2], ifma.IP[3] = 0, 0
			}
			ifmat = append(ifmat, ifma.toAddr())
		}
	}
	return ifmat, nil
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/routebsd"
	"syscall"
)

func interfaceMessages(ifindex int) ([]routebsd.Message, error) {
	rib, err := routebsd.FetchRIB(syscall.AF_UNSPEC, routebsd.RIBTypeInterface, ifindex)
	if err != nil {
		return nil, err
	}
	return routebsd.ParseRIB(routebsd.RIBTypeInterface, rib)
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	rib, err := routebsd.FetchRIB(syscall.AF_UNSPEC, syscall.NET_RT_IFMALIST, ifi.Index)
	if err != nil {
		return nil, err
	}
	msgs, err := routebsd.ParseRIB(syscall.NET_RT_IFMALIST, rib)
	if err != nil {
		return nil, err
	}
	ifmat := make([]Addr, 0, len(msgs))
	for _, m := range msgs {
		switch m := m.(type) {
		case *routebsd.InterfaceMulticastAddrMessage:
			if ifi.Index != m.Index {
				continue
			}
			var ip IP
			switch sa := m.Addrs[syscall.RTAX_IFA].(type) {
			case *routebsd.Inet4Addr:
				ip = IPv4(sa.IP[0], sa.IP[1], sa.IP[2], sa.IP[3])
			case *routebsd.Inet6Addr:
				ip = make(IP, IPv6len)
				copy(ip, sa.IP[:])
			}
			if ip != nil {
				ifmat = append(ifmat, &IPAddr{IP: ip})
			}
		}
	}
	return ifmat, nil
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"syscall"

	"golang.org/x/net/route"
)

func interfaceMessages(ifindex int) ([]route.Message, error) {
	typ := route.RIBType(syscall.NET_RT_IFLISTL)
	rib, err := route.FetchRIB(syscall.AF_UNSPEC, typ, ifindex)
	if err != nil {
		typ = route.RIBType(syscall.NET_RT_IFLIST)
		rib, err = route.FetchRIB(syscall.AF_UNSPEC, typ, ifindex)
		if err != nil {
			return nil, err
		}
	}
	return route.ParseRIB(typ, rib)
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	rib, err := route.FetchRIB(syscall.AF_UNSPEC, syscall.NET_RT_IFMALIST, ifi.Index)
	if err != nil {
		return nil, err
	}
	msgs, err := route.ParseRIB(syscall.NET_RT_IFMALIST, rib)
	if err != nil {
		return nil, err
	}
	ifmat := make([]Addr, 0, len(msgs))
	for _, m := range msgs {
		switch m := m.(type) {
		case *route.InterfaceMulticastAddrMessage:
			if ifi.Index != m.Index {
				continue
			}
			var ip IP
			switch sa := m.Addrs[syscall.RTAX_IFA].(type) {
			case *route.Inet4Addr:
				ip = IPv4(sa.IP[0], sa.IP[1], sa.IP[2], sa.IP[3])
			case *route.Inet6Addr:
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

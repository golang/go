// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/routebsd"
	"syscall"
)

func interfaceMessages(ifindex int) ([]routebsd.Message, error) {
	return routebsd.FetchRIBMessages(syscall.NET_RT_IFLIST, ifindex)
}

// interfaceMulticastAddrTable returns addresses for a specific
// interface.
func interfaceMulticastAddrTable(ifi *Interface) ([]Addr, error) {
	msgs, err := routebsd.FetchRIBMessages(syscall.NET_RT_IFMALIST, ifi.Index)
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
			case *routebsd.InetAddr:
				if sa.IP.Is4() {
					a := sa.IP.As4()
					ip = IPv4(a[0], a[1], a[2], a[3])
				} else if sa.IP.Is6() {
					a := sa.IP.As16()
					ip = make(IP, IPv6len)
					copy(ip, a[:])
				}
			}
			if ip != nil {
				ifmat = append(ifmat, &IPAddr{IP: ip})
			}
		}
	}
	return ifmat, nil
}

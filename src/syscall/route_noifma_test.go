// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build netbsd openbsd

package syscall_test

import (
	"fmt"
	"syscall"
)

func parseRoutingMessageHeader(m syscall.RoutingMessage) (addrFlags, error) {
	switch m := m.(type) {
	case *syscall.RouteMessage:
		errno := syscall.Errno(uintptr(m.Header.Errno))
		if errno != 0 {
			return 0, fmt.Errorf("%T: %v, %#v", m, errno, m.Header)
		}
		return addrFlags(m.Header.Addrs), nil
	case *syscall.InterfaceMessage:
		return addrFlags(m.Header.Addrs), nil
	case *syscall.InterfaceAddrMessage:
		return addrFlags(m.Header.Addrs), nil
	default:
		panic(fmt.Sprintf("unknown routing message type: %T", m))
	}
}

func parseRoutingSockaddrs(m syscall.RoutingMessage) ([]syscall.Sockaddr, error) {
	switch m := m.(type) {
	case *syscall.RouteMessage:
		sas, err := syscall.ParseRoutingSockaddr(m)
		if err != nil {
			return nil, fmt.Errorf("%T: %v, %#v", m, err, m.Data)
		}
		if err = sockaddrs(sas).match(addrFlags(m.Header.Addrs)); err != nil {
			return nil, err
		}
		return sas, nil
	case *syscall.InterfaceMessage:
		sas, err := syscall.ParseRoutingSockaddr(m)
		if err != nil {
			return nil, fmt.Errorf("%T: %v, %#v", m, err, m.Data)
		}
		if err = sockaddrs(sas).match(addrFlags(m.Header.Addrs)); err != nil {
			return nil, err
		}
		return sas, nil
	case *syscall.InterfaceAddrMessage:
		sas, err := syscall.ParseRoutingSockaddr(m)
		if err != nil {
			return nil, fmt.Errorf("%T: %v, %#v", m, err, m.Data)
		}
		if err = sockaddrs(sas).match(addrFlags(m.Header.Addrs)); err != nil {
			return nil, err
		}
		return sas, nil
	default:
		panic(fmt.Sprintf("unknown routing message type: %T", m))
	}
}

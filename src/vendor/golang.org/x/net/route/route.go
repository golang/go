// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd
// +build darwin dragonfly freebsd netbsd openbsd

// Package route provides basic functions for the manipulation of
// packet routing facilities on BSD variants.
//
// The package supports any version of Darwin, any version of
// DragonFly BSD, FreeBSD 7 and above, NetBSD 6 and above, and OpenBSD
// 5.6 and above.
package route

import (
	"errors"
	"os"
	"syscall"
)

var (
	errUnsupportedMessage = errors.New("unsupported message")
	errMessageMismatch    = errors.New("message mismatch")
	errMessageTooShort    = errors.New("message too short")
	errInvalidMessage     = errors.New("invalid message")
	errInvalidAddr        = errors.New("invalid address")
	errShortBuffer        = errors.New("short buffer")
)

// A RouteMessage represents a message conveying an address prefix, a
// nexthop address and an output interface.
//
// Unlike other messages, this message can be used to query adjacency
// information for the given address prefix, to add a new route, and
// to delete or modify the existing route from the routing information
// base inside the kernel by writing and reading route messages on a
// routing socket.
//
// For the manipulation of routing information, the route message must
// contain appropriate fields that include:
//
//	Version       = <must be specified>
//	Type          = <must be specified>
//	Flags         = <must be specified>
//	Index         = <must be specified if necessary>
//	ID            = <must be specified>
//	Seq           = <must be specified>
//	Addrs         = <must be specified>
//
// The Type field specifies a type of manipulation, the Flags field
// specifies a class of target information and the Addrs field
// specifies target information like the following:
//
//	route.RouteMessage{
//		Version: RTM_VERSION,
//		Type: RTM_GET,
//		Flags: RTF_UP | RTF_HOST,
//		ID: uintptr(os.Getpid()),
//		Seq: 1,
//		Addrs: []route.Addrs{
//			RTAX_DST: &route.Inet4Addr{ ... },
//			RTAX_IFP: &route.LinkAddr{ ... },
//			RTAX_BRD: &route.Inet4Addr{ ... },
//		},
//	}
//
// The values for the above fields depend on the implementation of
// each operating system.
//
// The Err field on a response message contains an error value on the
// requested operation. If non-nil, the requested operation is failed.
type RouteMessage struct {
	Version int     // message version
	Type    int     // message type
	Flags   int     // route flags
	Index   int     // interface index when attached
	ID      uintptr // sender's identifier; usually process ID
	Seq     int     // sequence number
	Err     error   // error on requested operation
	Addrs   []Addr  // addresses

	extOff int    // offset of header extension
	raw    []byte // raw message
}

// Marshal returns the binary encoding of m.
func (m *RouteMessage) Marshal() ([]byte, error) {
	return m.marshal()
}

// A RIBType represents a type of routing information base.
type RIBType int

const (
	RIBTypeRoute     RIBType = syscall.NET_RT_DUMP
	RIBTypeInterface RIBType = syscall.NET_RT_IFLIST
)

// FetchRIB fetches a routing information base from the operating
// system.
//
// The provided af must be an address family.
//
// The provided arg must be a RIBType-specific argument.
// When RIBType is related to routes, arg might be a set of route
// flags. When RIBType is related to network interfaces, arg might be
// an interface index or a set of interface flags. In most cases, zero
// means a wildcard.
func FetchRIB(af int, typ RIBType, arg int) ([]byte, error) {
	try := 0
	for {
		try++
		mib := [6]int32{syscall.CTL_NET, syscall.AF_ROUTE, 0, int32(af), int32(typ), int32(arg)}
		n := uintptr(0)
		if err := sysctl(mib[:], nil, &n, nil, 0); err != nil {
			return nil, os.NewSyscallError("sysctl", err)
		}
		if n == 0 {
			return nil, nil
		}
		b := make([]byte, n)
		if err := sysctl(mib[:], &b[0], &n, nil, 0); err != nil {
			// If the sysctl failed because the data got larger
			// between the two sysctl calls, try a few times
			// before failing. (golang.org/issue/45736).
			const maxTries = 3
			if err == syscall.ENOMEM && try < maxTries {
				continue
			}
			return nil, os.NewSyscallError("sysctl", err)
		}
		return b[:n], nil
	}
}

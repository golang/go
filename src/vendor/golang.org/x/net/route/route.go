// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

// Package route provides basic functions for the manipulation of
// packet routing facilities on BSD variants.
//
// The package supports any version of Darwin, any version of
// DragonFly BSD, FreeBSD 7 through 11, NetBSD 6 and above, and
// OpenBSD 5.6 and above.
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
)

// A RouteMessage represents a message conveying an address prefix, a
// nexthop address and an output interface.
type RouteMessage struct {
	Version int    // message version
	Type    int    // message type
	Flags   int    // route flags
	Index   int    // interface index when atatched
	Addrs   []Addr // addresses

	extOff int    // offset of header extension
	raw    []byte // raw message
}

// A RIBType reprensents a type of routing information base.
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
	mib := [6]int32{sysCTL_NET, sysAF_ROUTE, 0, int32(af), int32(typ), int32(arg)}
	n := uintptr(0)
	if err := sysctl(mib[:], nil, &n, nil, 0); err != nil {
		return nil, os.NewSyscallError("sysctl", err)
	}
	if n == 0 {
		return nil, nil
	}
	b := make([]byte, n)
	if err := sysctl(mib[:], &b[0], &n, nil, 0); err != nil {
		return nil, os.NewSyscallError("sysctl", err)
	}
	return b[:n], nil
}

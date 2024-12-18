// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

// Package routebsd supports reading interface addresses on BSD systems.
// This is a very stripped down version of x/net/route,
// for use by the net package in the standard library.
package routebsd

import (
	"errors"
	"syscall"
)

var (
	errMessageMismatch = errors.New("message mismatch")
	errMessageTooShort = errors.New("message too short")
	errInvalidMessage  = errors.New("invalid message")
	errInvalidAddr     = errors.New("invalid address")
)

// fetchRIB fetches a routing information base from the operating
// system.
//
// The arg is an interface index or 0 for all.
func fetchRIB(typ, arg int) ([]byte, error) {
	try := 0
	for {
		try++
		b, err := syscall.RouteRIB(typ, arg)

		// If the sysctl failed because the data got larger
		// between the two sysctl calls, try a few times
		// before failing (issue #45736).
		const maxTries = 3
		if err == syscall.ENOMEM && try < maxTries {
			continue
		}

		return b, err
	}
}

// FetchRIBMessages fetches a list of addressing messages for an interface.
// The typ argument is something like syscall.NET_RT_IFLIST.
// The argument is an interface index or 0 for all.
func FetchRIBMessages(typ, arg int) ([]Message, error) {
	b, err := fetchRIB(typ, arg)
	if err != nil {
		return nil, err
	}
	ms, err := parseRIB(b)
	if err != nil {
		return nil, err
	}
	return ms, nil
}

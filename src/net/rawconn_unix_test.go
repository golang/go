// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix darwin dragonfly freebsd linux netbsd openbsd solaris

package net

import (
	"errors"
	"syscall"
)

func readRawConn(c syscall.RawConn, b []byte) (int, error) {
	var operr error
	var n int
	err := c.Read(func(s uintptr) bool {
		n, operr = syscall.Read(int(s), b)
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	})
	if err != nil {
		return n, err
	}
	if operr != nil {
		return n, operr
	}
	return n, nil
}

func writeRawConn(c syscall.RawConn, b []byte) error {
	var operr error
	err := c.Write(func(s uintptr) bool {
		_, operr = syscall.Write(int(s), b)
		if operr == syscall.EAGAIN {
			return false
		}
		return true
	})
	if err != nil {
		return err
	}
	if operr != nil {
		return operr
	}
	return nil
}

func controlRawConn(c syscall.RawConn, addr Addr) error {
	var operr error
	fn := func(s uintptr) {
		_, operr = syscall.GetsockoptInt(int(s), syscall.SOL_SOCKET, syscall.SO_REUSEADDR)
		if operr != nil {
			return
		}
		switch addr := addr.(type) {
		case *TCPAddr:
			// There's no guarantee that IP-level socket
			// options work well with dual stack sockets.
			// A simple solution would be to take a look
			// at the bound address to the raw connection
			// and to classify the address family of the
			// underlying socket by the bound address:
			//
			// - When IP.To16() != nil and IP.To4() == nil,
			//   we can assume that the raw connection
			//   consists of an IPv6 socket using only
			//   IPv6 addresses.
			//
			// - When IP.To16() == nil and IP.To4() != nil,
			//   the raw connection consists of an IPv4
			//   socket using only IPv4 addresses.
			//
			// - Otherwise, the raw connection is a dual
			//   stack socket, an IPv6 socket using IPv6
			//   addresses including IPv4-mapped or
			//   IPv4-embedded IPv6 addresses.
			if addr.IP.To16() != nil && addr.IP.To4() == nil {
				operr = syscall.SetsockoptInt(int(s), syscall.IPPROTO_IPV6, syscall.IPV6_UNICAST_HOPS, 1)
			} else if addr.IP.To16() == nil && addr.IP.To4() != nil {
				operr = syscall.SetsockoptInt(int(s), syscall.IPPROTO_IP, syscall.IP_TTL, 1)
			}
		}
	}
	if err := c.Control(fn); err != nil {
		return err
	}
	if operr != nil {
		return operr
	}
	return nil
}

func controlOnConnSetup(network string, address string, c syscall.RawConn) error {
	var operr error
	var fn func(uintptr)
	switch network {
	case "tcp", "udp", "ip":
		return errors.New("ambiguous network: " + network)
	case "unix", "unixpacket", "unixgram":
		fn = func(s uintptr) {
			_, operr = syscall.GetsockoptInt(int(s), syscall.SOL_SOCKET, syscall.SO_ERROR)
		}
	default:
		switch network[len(network)-1] {
		case '4':
			fn = func(s uintptr) {
				operr = syscall.SetsockoptInt(int(s), syscall.IPPROTO_IP, syscall.IP_TTL, 1)
			}
		case '6':
			fn = func(s uintptr) {
				operr = syscall.SetsockoptInt(int(s), syscall.IPPROTO_IPV6, syscall.IPV6_UNICAST_HOPS, 1)
			}
		default:
			return errors.New("unknown network: " + network)
		}
	}
	if err := c.Control(fn); err != nil {
		return err
	}
	if operr != nil {
		return operr
	}
	return nil
}

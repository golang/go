// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"syscall"
	"unsafe"
)

func readRawConn(c syscall.RawConn, b []byte) (int, error) {
	var operr error
	var n int
	err := c.Read(func(s uintptr) bool {
		var read uint32
		var flags uint32
		var buf syscall.WSABuf
		buf.Buf = &b[0]
		buf.Len = uint32(len(b))
		operr = syscall.WSARecv(syscall.Handle(s), &buf, 1, &read, &flags, nil, nil)
		n = int(read)
		return true
	})
	if err != nil {
		return n, err
	}
	return n, operr
}

func writeRawConn(c syscall.RawConn, b []byte) error {
	var operr error
	err := c.Write(func(s uintptr) bool {
		var written uint32
		var buf syscall.WSABuf
		buf.Buf = &b[0]
		buf.Len = uint32(len(b))
		operr = syscall.WSASend(syscall.Handle(s), &buf, 1, &written, 0, nil, nil)
		return true
	})
	if err != nil {
		return err
	}
	return operr
}

func controlRawConn(c syscall.RawConn, addr Addr) error {
	var operr error
	fn := func(s uintptr) {
		var v, l int32
		l = int32(unsafe.Sizeof(v))
		operr = syscall.Getsockopt(syscall.Handle(s), syscall.SOL_SOCKET, syscall.SO_REUSEADDR, (*byte)(unsafe.Pointer(&v)), &l)
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
				operr = syscall.SetsockoptInt(syscall.Handle(s), syscall.IPPROTO_IPV6, syscall.IPV6_UNICAST_HOPS, 1)
			} else if addr.IP.To16() == nil && addr.IP.To4() != nil {
				operr = syscall.SetsockoptInt(syscall.Handle(s), syscall.IPPROTO_IP, syscall.IP_TTL, 1)
			}
		}
	}
	if err := c.Control(fn); err != nil {
		return err
	}
	return operr
}

func controlOnConnSetup(network string, address string, c syscall.RawConn) error {
	var operr error
	var fn func(uintptr)
	switch network {
	case "tcp", "udp", "ip":
		return errors.New("ambiguous network: " + network)
	case "unix", "unixpacket", "unixgram":
		fn = func(s uintptr) {
			const SO_ERROR = 0x1007
			_, operr = syscall.GetsockoptInt(syscall.Handle(s), syscall.SOL_SOCKET, SO_ERROR)
		}
	default:
		switch network[len(network)-1] {
		case '4':
			fn = func(s uintptr) {
				operr = syscall.SetsockoptInt(syscall.Handle(s), syscall.IPPROTO_IP, syscall.IP_TTL, 1)
			}
		case '6':
			fn = func(s uintptr) {
				operr = syscall.SetsockoptInt(syscall.Handle(s), syscall.IPPROTO_IPV6, syscall.IPV6_UNICAST_HOPS, 1)
			}
		default:
			return errors.New("unknown network: " + network)
		}
	}
	if err := c.Control(fn); err != nil {
		return err
	}
	return operr
}

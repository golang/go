// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "os"

// Dial connects to the remote address raddr on the network net.
// If the string laddr is not empty, it is used as the local address
// for the connection.
//
// Known networks are "tcp", "tcp4" (IPv4-only), "tcp6" (IPv6-only),
// "udp", "udp4" (IPv4-only), "udp6" (IPv6-only), "ip", "ip4"
// (IPv4-only), "ip6" (IPv6-only), "unix" and "unixgram".
//
// For IP networks, addresses have the form host:port.  If host is
// a literal IPv6 address, it must be enclosed in square brackets.
//
// Examples:
//	Dial("tcp", "", "12.34.56.78:80")
//	Dial("tcp", "", "google.com:80")
//	Dial("tcp", "", "[de:ad:be:ef::ca:fe]:80")
//	Dial("tcp", "127.0.0.1:123", "127.0.0.1:88")
//
func Dial(net, laddr, raddr string) (c Conn, err os.Error) {
	switch prefixBefore(net, ':') {
	case "tcp", "tcp4", "tcp6":
		var la, ra *TCPAddr
		if laddr != "" {
			if la, err = ResolveTCPAddr(laddr); err != nil {
				goto Error
			}
		}
		if raddr != "" {
			if ra, err = ResolveTCPAddr(raddr); err != nil {
				goto Error
			}
		}
		c, err := DialTCP(net, la, ra)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "udp", "udp4", "udp6":
		var la, ra *UDPAddr
		if laddr != "" {
			if la, err = ResolveUDPAddr(laddr); err != nil {
				goto Error
			}
		}
		if raddr != "" {
			if ra, err = ResolveUDPAddr(raddr); err != nil {
				goto Error
			}
		}
		c, err := DialUDP(net, la, ra)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "unix", "unixgram", "unixpacket":
		var la, ra *UnixAddr
		if raddr != "" {
			if ra, err = ResolveUnixAddr(net, raddr); err != nil {
				goto Error
			}
		}
		if laddr != "" {
			if la, err = ResolveUnixAddr(net, laddr); err != nil {
				goto Error
			}
		}
		c, err = DialUnix(net, la, ra)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "ip", "ip4", "ip6":
		var la, ra *IPAddr
		if laddr != "" {
			if la, err = ResolveIPAddr(laddr); err != nil {
				goto Error
			}
		}
		if raddr != "" {
			if ra, err = ResolveIPAddr(raddr); err != nil {
				goto Error
			}
		}
		c, err := DialIP(net, la, ra)
		if err != nil {
			return nil, err
		}
		return c, nil

	}
	err = UnknownNetworkError(net)
Error:
	return nil, &OpError{"dial", net + " " + raddr, nil, err}
}

// Listen announces on the local network address laddr.
// The network string net must be a stream-oriented
// network: "tcp", "tcp4", "tcp6", or "unix", or "unixpacket".
func Listen(net, laddr string) (l Listener, err os.Error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
		var la *TCPAddr
		if laddr != "" {
			if la, err = ResolveTCPAddr(laddr); err != nil {
				return nil, err
			}
		}
		l, err := ListenTCP(net, la)
		if err != nil {
			return nil, err
		}
		return l, nil
	case "unix", "unixpacket":
		var la *UnixAddr
		if laddr != "" {
			if la, err = ResolveUnixAddr(net, laddr); err != nil {
				return nil, err
			}
		}
		l, err := ListenUnix(net, la)
		if err != nil {
			return nil, err
		}
		return l, nil
	}
	return nil, UnknownNetworkError(net)
}

// ListenPacket announces on the local network address laddr.
// The network string net must be a packet-oriented network:
// "udp", "udp4", "udp6", or "unixgram".
func ListenPacket(net, laddr string) (c PacketConn, err os.Error) {
	switch prefixBefore(net, ':') {
	case "udp", "udp4", "udp6":
		var la *UDPAddr
		if laddr != "" {
			if la, err = ResolveUDPAddr(laddr); err != nil {
				return nil, err
			}
		}
		c, err := ListenUDP(net, la)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "unixgram":
		var la *UnixAddr
		if laddr != "" {
			if la, err = ResolveUnixAddr(net, laddr); err != nil {
				return nil, err
			}
		}
		c, err := DialUnix(net, la, nil)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "ip", "ip4", "ip6":
		var la *IPAddr
		if laddr != "" {
			if la, err = ResolveIPAddr(laddr); err != nil {
				return nil, err
			}
		}
		c, err := ListenIP(net, la)
		if err != nil {
			return nil, err
		}
		return c, nil
	}
	return nil, UnknownNetworkError(net)
}

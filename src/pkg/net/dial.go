// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import "os"

// Dial connects to the address addr on the network net.
//
// Known networks are "tcp", "tcp4" (IPv4-only), "tcp6" (IPv6-only),
// "udp", "udp4" (IPv4-only), "udp6" (IPv6-only), "ip", "ip4"
// (IPv4-only), "ip6" (IPv6-only), "unix" and "unixgram".
//
// For IP networks, addresses have the form host:port.  If host is
// a literal IPv6 address, it must be enclosed in square brackets.
// The functions JoinHostPort and SplitHostPort manipulate 
// addresses in this form.
//
// Examples:
//	Dial("tcp", "12.34.56.78:80")
//	Dial("tcp", "google.com:80")
//	Dial("tcp", "[de:ad:be:ef::ca:fe]:80")
//
func Dial(net, addr string) (c Conn, err os.Error) {
	raddr := addr
	if raddr == "" {
		return nil, &OpError{"dial", net, nil, errMissingAddress}
	}
	switch net {
	case "tcp", "tcp4", "tcp6":
		var ra *TCPAddr
		if ra, err = ResolveTCPAddr(net, raddr); err != nil {
			goto Error
		}
		c, err := DialTCP(net, nil, ra)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "udp", "udp4", "udp6":
		var ra *UDPAddr
		if ra, err = ResolveUDPAddr(net, raddr); err != nil {
			goto Error
		}
		c, err := DialUDP(net, nil, ra)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "unix", "unixgram", "unixpacket":
		var ra *UnixAddr
		if ra, err = ResolveUnixAddr(net, raddr); err != nil {
			goto Error
		}
		c, err = DialUnix(net, nil, ra)
		if err != nil {
			return nil, err
		}
		return c, nil
	case "ip", "ip4", "ip6":
		var ra *IPAddr
		if ra, err = ResolveIPAddr(raddr); err != nil {
			goto Error
		}
		c, err := DialIP(net, nil, ra)
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
			if la, err = ResolveTCPAddr(net, laddr); err != nil {
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
	switch net {
	case "udp", "udp4", "udp6":
		var la *UDPAddr
		if laddr != "" {
			if la, err = ResolveUDPAddr(net, laddr); err != nil {
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
	}

	if i := last(net, ':'); i > 0 {
		switch net[0:i] {
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
	}

	return nil, UnknownNetworkError(net)
}

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"time"
)

func resolveNetAddr(op, net, addr string) (a Addr, err error) {
	if addr == "" {
		return nil, &OpError{op, net, nil, errMissingAddress}
	}
	switch net {
	case "tcp", "tcp4", "tcp6":
		a, err = ResolveTCPAddr(net, addr)
	case "udp", "udp4", "udp6":
		a, err = ResolveUDPAddr(net, addr)
	case "unix", "unixgram", "unixpacket":
		a, err = ResolveUnixAddr(net, addr)
	case "ip", "ip4", "ip6":
		a, err = ResolveIPAddr(net, addr)
	default:
		err = UnknownNetworkError(net)
	}
	if err != nil {
		return nil, &OpError{op, net + " " + addr, nil, err}
	}
	return
}

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
func Dial(net, addr string) (Conn, error) {
	addri, err := resolveNetAddr("dial", net, addr)
	if err != nil {
		return nil, err
	}
	return dialAddr(net, addr, addri)
}

func dialAddr(net, addr string, addri Addr) (c Conn, err error) {
	switch ra := addri.(type) {
	case *TCPAddr:
		c, err = DialTCP(net, nil, ra)
	case *UDPAddr:
		c, err = DialUDP(net, nil, ra)
	case *UnixAddr:
		c, err = DialUnix(net, nil, ra)
	case *IPAddr:
		c, err = DialIP(net, nil, ra)
	default:
		err = &OpError{"dial", net + " " + addr, nil, UnknownNetworkError(net)}
	}
	if err != nil {
		return nil, err
	}
	return
}

// DialTimeout acts like Dial but takes a timeout.
// The timeout includes name resolution, if required.
func DialTimeout(net, addr string, timeout time.Duration) (Conn, error) {
	// TODO(bradfitz): the timeout should be pushed down into the
	// net package's event loop, so on timeout to dead hosts we
	// don't have a goroutine sticking around for the default of
	// ~3 minutes.
	t := time.NewTimer(timeout)
	defer t.Stop()
	type pair struct {
		Conn
		error
	}
	ch := make(chan pair, 1)
	resolvedAddr := make(chan Addr, 1)
	go func() {
		addri, err := resolveNetAddr("dial", net, addr)
		if err != nil {
			ch <- pair{nil, err}
			return
		}
		resolvedAddr <- addri // in case we need it for OpError
		c, err := dialAddr(net, addr, addri)
		ch <- pair{c, err}
	}()
	select {
	case <-t.C:
		// Try to use the real Addr in our OpError, if we resolved it
		// before the timeout. Otherwise we just use stringAddr.
		var addri Addr
		select {
		case a := <-resolvedAddr:
			addri = a
		default:
			addri = &stringAddr{net, addr}
		}
		err := &OpError{
			Op:   "dial",
			Net:  net,
			Addr: addri,
			Err:  &timeoutError{},
		}
		return nil, err
	case p := <-ch:
		return p.Conn, p.error
	}
	panic("unreachable")
}

type stringAddr struct {
	net, addr string
}

func (a stringAddr) Network() string { return a.net }
func (a stringAddr) String() string  { return a.addr }

// Listen announces on the local network address laddr.
// The network string net must be a stream-oriented
// network: "tcp", "tcp4", "tcp6", or "unix", or "unixpacket".
func Listen(net, laddr string) (l Listener, err error) {
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
func ListenPacket(net, laddr string) (c PacketConn, err error) {
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

	var rawnet string
	if rawnet, _, err = splitNetProto(net); err != nil {
		switch rawnet {
		case "ip", "ip4", "ip6":
			var la *IPAddr
			if laddr != "" {
				if la, err = ResolveIPAddr(rawnet, laddr); err != nil {
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

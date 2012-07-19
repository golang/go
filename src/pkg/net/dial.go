// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"time"
)

func parseDialNetwork(net string) (afnet string, proto int, err error) {
	i := last(net, ':')
	if i < 0 { // no colon
		switch net {
		case "tcp", "tcp4", "tcp6":
		case "udp", "udp4", "udp6":
		case "unix", "unixgram", "unixpacket":
		default:
			return "", 0, UnknownNetworkError(net)
		}
		return net, 0, nil
	}
	afnet = net[:i]
	switch afnet {
	case "ip", "ip4", "ip6":
		protostr := net[i+1:]
		proto, i, ok := dtoi(protostr, 0)
		if !ok || i != len(protostr) {
			proto, err = lookupProtocol(protostr)
			if err != nil {
				return "", 0, err
			}
		}
		return afnet, proto, nil
	}
	return "", 0, UnknownNetworkError(net)
}

func resolveNetAddr(op, net, addr string) (afnet string, a Addr, err error) {
	afnet, _, err = parseDialNetwork(net)
	if err != nil {
		return "", nil, &OpError{op, net, nil, err}
	}
	if op == "dial" && addr == "" {
		return "", nil, &OpError{op, net, nil, errMissingAddress}
	}
	switch afnet {
	case "tcp", "tcp4", "tcp6":
		if addr != "" {
			a, err = ResolveTCPAddr(afnet, addr)
		}
	case "udp", "udp4", "udp6":
		if addr != "" {
			a, err = ResolveUDPAddr(afnet, addr)
		}
	case "ip", "ip4", "ip6":
		if addr != "" {
			a, err = ResolveIPAddr(afnet, addr)
		}
	case "unix", "unixgram", "unixpacket":
		if addr != "" {
			a, err = ResolveUnixAddr(afnet, addr)
		}
	}
	return
}

// Dial connects to the address addr on the network net.
//
// Known networks are "tcp", "tcp4" (IPv4-only), "tcp6" (IPv6-only),
// "udp", "udp4" (IPv4-only), "udp6" (IPv6-only), "ip", "ip4"
// (IPv4-only), "ip6" (IPv6-only), "unix" and "unixpacket".
//
// For TCP and UDP networks, addresses have the form host:port.
// If host is a literal IPv6 address, it must be enclosed
// in square brackets.  The functions JoinHostPort and SplitHostPort
// manipulate addresses in this form.
//
// Examples:
//	Dial("tcp", "12.34.56.78:80")
//	Dial("tcp", "google.com:80")
//	Dial("tcp", "[de:ad:be:ef::ca:fe]:80")
//
// For IP networks, addr must be "ip", "ip4" or "ip6" followed
// by a colon and a protocol number or name.
//
// Examples:
//	Dial("ip4:1", "127.0.0.1")
//	Dial("ip6:ospf", "::1")
//
func Dial(net, addr string) (Conn, error) {
	_, addri, err := resolveNetAddr("dial", net, addr)
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
	case *IPAddr:
		c, err = DialIP(net, nil, ra)
	case *UnixAddr:
		c, err = DialUnix(net, nil, ra)
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
		_, addri, err := resolveNetAddr("dial", net, addr)
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
// The network string net must be a stream-oriented network:
// "tcp", "tcp4", "tcp6", "unix" or "unixpacket".
func Listen(net, laddr string) (Listener, error) {
	afnet, a, err := resolveNetAddr("listen", net, laddr)
	if err != nil {
		return nil, err
	}
	switch afnet {
	case "tcp", "tcp4", "tcp6":
		var la *TCPAddr
		if a != nil {
			la = a.(*TCPAddr)
		}
		return ListenTCP(net, la)
	case "unix", "unixpacket":
		var la *UnixAddr
		if a != nil {
			la = a.(*UnixAddr)
		}
		return ListenUnix(net, la)
	}
	return nil, UnknownNetworkError(net)
}

// ListenPacket announces on the local network address laddr.
// The network string net must be a packet-oriented network:
// "udp", "udp4", "udp6", "ip", "ip4", "ip6" or "unixgram".
func ListenPacket(net, addr string) (PacketConn, error) {
	afnet, a, err := resolveNetAddr("listen", net, addr)
	if err != nil {
		return nil, err
	}
	switch afnet {
	case "udp", "udp4", "udp6":
		var la *UDPAddr
		if a != nil {
			la = a.(*UDPAddr)
		}
		return ListenUDP(net, la)
	case "ip", "ip4", "ip6":
		var la *IPAddr
		if a != nil {
			la = a.(*IPAddr)
		}
		return ListenIP(net, la)
	case "unixgram":
		var la *UnixAddr
		if a != nil {
			la = a.(*UnixAddr)
		}
		return DialUnix(net, la, nil)
	}
	return nil, UnknownNetworkError(net)
}

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"time"
)

// A DialOption modifies a DialOpt call.
type DialOption interface {
	setDialOpt(*dialOpts)
}

var noLocalAddr Addr // nil

// dialOpts holds all the dial options, populated by a DialOption's
// setDialOpt.
//
// All fields may be their zero value.
type dialOpts struct {
	deadline        time.Time
	localAddr       Addr
	network         string // if empty, "tcp"
	deferredConnect bool
}

func (o *dialOpts) net() string {
	if o.network == "" {
		return "tcp"
	}
	return o.network
}

var (
	// TCP is a dial option to dial with TCP (over IPv4 or IPv6).
	TCP = Network("tcp")

	// UDP is a dial option to dial with UDP (over IPv4 or IPv6).
	UDP = Network("udp")
)

// Network returns a DialOption to dial using the given network.
//
// Known networks are "tcp", "tcp4" (IPv4-only), "tcp6" (IPv6-only),
// "udp", "udp4" (IPv4-only), "udp6" (IPv6-only), "ip", "ip4"
// (IPv4-only), "ip6" (IPv6-only), "unix", "unixgram" and
// "unixpacket".
//
// For IP networks, net must be "ip", "ip4" or "ip6" followed
// by a colon and a protocol number or name, such as
// "ipv4:1" or "ip6:ospf".
func Network(net string) DialOption {
	return dialNetwork(net)
}

type dialNetwork string

func (s dialNetwork) setDialOpt(o *dialOpts) {
	o.network = string(s)
}

// Deadline returns a DialOption to fail a dial that doesn't
// complete before t.
func Deadline(t time.Time) DialOption {
	return dialDeadline(t)
}

type dialDeadline time.Time

func (t dialDeadline) setDialOpt(o *dialOpts) {
	o.deadline = time.Time(t)
}

// Timeout returns a DialOption to fail a dial that doesn't
// complete within the provided duration.
func Timeout(d time.Duration) DialOption {
	return dialTimeoutOpt(d)
}

type dialTimeoutOpt time.Duration

func (d dialTimeoutOpt) setDialOpt(o *dialOpts) {
	o.deadline = time.Now().Add(time.Duration(d))
}

type tcpFastOpen struct{}

func (tcpFastOpen) setDialOpt(o *dialOpts) {
	o.deferredConnect = true
}

// TODO(bradfitz): implement this (golang.org/issue/4842) and unexport this.
//
// TCPFastTimeout returns an option to use TCP Fast Open (TFO) when
// doing this dial. It is only valid for use with TCP connections.
// Data sent over a TFO connection may be processed by the peer
// multiple times, so should be used with caution.
func todo_TCPFastTimeout() DialOption {
	return tcpFastOpen{}
}

type localAddrOption struct {
	la Addr
}

func (a localAddrOption) setDialOpt(o *dialOpts) {
	o.localAddr = a.la
}

// LocalAddress returns a dial option to perform a dial with the
// provided local address. The address must be of a compatible type
// for the network being dialed.
func LocalAddress(addr Addr) DialOption {
	return localAddrOption{addr}
}

func parseNetwork(net string) (afnet string, proto int, err error) {
	i := last(net, ':')
	if i < 0 { // no colon
		switch net {
		case "tcp", "tcp4", "tcp6":
		case "udp", "udp4", "udp6":
		case "ip", "ip4", "ip6":
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

func resolveAddr(op, net, addr string, deadline time.Time) (Addr, error) {
	afnet, _, err := parseNetwork(net)
	if err != nil {
		return nil, &OpError{op, net, nil, err}
	}
	if op == "dial" && addr == "" {
		return nil, &OpError{op, net, nil, errMissingAddress}
	}
	switch afnet {
	case "unix", "unixgram", "unixpacket":
		return ResolveUnixAddr(afnet, addr)
	}
	return resolveInternetAddr(afnet, addr, deadline)
}

// Dial connects to the address addr on the network net.
//
// Known networks are "tcp", "tcp4" (IPv4-only), "tcp6" (IPv6-only),
// "udp", "udp4" (IPv4-only), "udp6" (IPv6-only), "ip", "ip4"
// (IPv4-only), "ip6" (IPv6-only), "unix", "unixgram" and
// "unixpacket".
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
// For IP networks, net must be "ip", "ip4" or "ip6" followed
// by a colon and a protocol number or name.
//
// Examples:
//	Dial("ip4:1", "127.0.0.1")
//	Dial("ip6:ospf", "::1")
//
func Dial(net, addr string) (Conn, error) {
	return DialOpt(addr, dialNetwork(net))
}

// DialOpt dials addr using the provided options.
// If no options are provided, DialOpt(addr) is equivalent
// to Dial("tcp", addr). See Dial for the syntax of addr.
func DialOpt(addr string, opts ...DialOption) (Conn, error) {
	var o dialOpts
	for _, opt := range opts {
		opt.setDialOpt(&o)
	}
	ra, err := resolveAddr("dial", o.net(), addr, o.deadline)
	if err != nil {
		return nil, err
	}
	return dial(o.net(), addr, o.localAddr, ra, o.deadline)
}

func dial(net, addr string, la, ra Addr, deadline time.Time) (c Conn, err error) {
	if la != nil && la.Network() != ra.Network() {
		return nil, &OpError{"dial", net, ra, errors.New("mismatched local addr type " + la.Network())}
	}
	switch ra := ra.(type) {
	case *TCPAddr:
		la, _ := la.(*TCPAddr)
		c, err = dialTCP(net, la, ra, deadline)
	case *UDPAddr:
		la, _ := la.(*UDPAddr)
		c, err = dialUDP(net, la, ra, deadline)
	case *IPAddr:
		la, _ := la.(*IPAddr)
		c, err = dialIP(net, la, ra, deadline)
	case *UnixAddr:
		la, _ := la.(*UnixAddr)
		c, err = dialUnix(net, la, ra, deadline)
	default:
		err = &OpError{"dial", net + " " + addr, ra, UnknownNetworkError(net)}
	}
	if err != nil {
		return nil, err
	}
	return
}

// DialTimeout acts like Dial but takes a timeout.
// The timeout includes name resolution, if required.
func DialTimeout(net, addr string, timeout time.Duration) (Conn, error) {
	return dialTimeout(net, addr, timeout)
}

// dialTimeoutRace is the old implementation of DialTimeout, still used
// on operating systems where the deadline hasn't been pushed down
// into the pollserver.
// TODO: fix this on plan9.
func dialTimeoutRace(net, addr string, timeout time.Duration) (Conn, error) {
	t := time.NewTimer(timeout)
	defer t.Stop()
	type pair struct {
		Conn
		error
	}
	ch := make(chan pair, 1)
	resolvedAddr := make(chan Addr, 1)
	go func() {
		ra, err := resolveAddr("dial", net, addr, noDeadline)
		if err != nil {
			ch <- pair{nil, err}
			return
		}
		resolvedAddr <- ra // in case we need it for OpError
		c, err := dial(net, addr, noLocalAddr, ra, noDeadline)
		ch <- pair{c, err}
	}()
	select {
	case <-t.C:
		// Try to use the real Addr in our OpError, if we resolved it
		// before the timeout. Otherwise we just use stringAddr.
		var ra Addr
		select {
		case a := <-resolvedAddr:
			ra = a
		default:
			ra = &stringAddr{net, addr}
		}
		err := &OpError{
			Op:   "dial",
			Net:  net,
			Addr: ra,
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
	la, err := resolveAddr("listen", net, laddr, noDeadline)
	if err != nil {
		return nil, err
	}
	switch la := la.(type) {
	case *TCPAddr:
		return ListenTCP(net, la)
	case *UnixAddr:
		return ListenUnix(net, la)
	}
	return nil, UnknownNetworkError(net)
}

// ListenPacket announces on the local network address laddr.
// The network string net must be a packet-oriented network:
// "udp", "udp4", "udp6", "ip", "ip4", "ip6" or "unixgram".
func ListenPacket(net, laddr string) (PacketConn, error) {
	la, err := resolveAddr("listen", net, laddr, noDeadline)
	if err != nil {
		return nil, err
	}
	switch la := la.(type) {
	case *UDPAddr:
		return ListenUDP(net, la)
	case *IPAddr:
		return ListenIP(net, la)
	case *UnixAddr:
		return ListenUnixgram(net, la)
	}
	return nil, UnknownNetworkError(net)
}

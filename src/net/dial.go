// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"errors"
	"time"
)

// A Dialer contains options for connecting to an address.
//
// The zero value for each field is equivalent to dialing
// without that option. Dialing with the zero value of Dialer
// is therefore equivalent to just calling the Dial function.
type Dialer struct {
	// Timeout is the maximum amount of time a dial will wait for
	// a connect to complete. If Deadline is also set, it may fail
	// earlier.
	//
	// The default is no timeout.
	//
	// With or without a timeout, the operating system may impose
	// its own earlier timeout. For instance, TCP timeouts are
	// often around 3 minutes.
	Timeout time.Duration

	// Deadline is the absolute point in time after which dials
	// will fail. If Timeout is set, it may fail earlier.
	// Zero means no deadline, or dependent on the operating system
	// as with the Timeout option.
	Deadline time.Time

	// LocalAddr is the local address to use when dialing an
	// address. The address must be of a compatible type for the
	// network being dialed.
	// If nil, a local address is automatically chosen.
	LocalAddr Addr

	// DualStack allows a single dial to attempt to establish
	// multiple IPv4 and IPv6 connections and to return the first
	// established connection when the network is "tcp" and the
	// destination is a host name that has multiple address family
	// DNS records.
	DualStack bool

	// KeepAlive specifies the keep-alive period for an active
	// network connection.
	// If zero, keep-alives are not enabled. Network protocols
	// that do not support keep-alives ignore this field.
	KeepAlive time.Duration
}

// Return either now+Timeout or Deadline, whichever comes first.
// Or zero, if neither is set.
func (d *Dialer) deadline() time.Time {
	if d.Timeout == 0 {
		return d.Deadline
	}
	timeoutDeadline := time.Now().Add(d.Timeout)
	if d.Deadline.IsZero() || timeoutDeadline.Before(d.Deadline) {
		return timeoutDeadline
	} else {
		return d.Deadline
	}
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

func resolveAddr(op, net, addr string, deadline time.Time) (netaddr, error) {
	afnet, _, err := parseNetwork(net)
	if err != nil {
		return nil, err
	}
	if op == "dial" && addr == "" {
		return nil, errMissingAddress
	}
	switch afnet {
	case "unix", "unixgram", "unixpacket":
		return ResolveUnixAddr(afnet, addr)
	}
	return resolveInternetAddr(afnet, addr, deadline)
}

// Dial connects to the address on the named network.
//
// Known networks are "tcp", "tcp4" (IPv4-only), "tcp6" (IPv6-only),
// "udp", "udp4" (IPv4-only), "udp6" (IPv6-only), "ip", "ip4"
// (IPv4-only), "ip6" (IPv6-only), "unix", "unixgram" and
// "unixpacket".
//
// For TCP and UDP networks, addresses have the form host:port.
// If host is a literal IPv6 address it must be enclosed
// in square brackets as in "[::1]:80" or "[ipv6-host%zone]:80".
// The functions JoinHostPort and SplitHostPort manipulate addresses
// in this form.
//
// Examples:
//	Dial("tcp", "12.34.56.78:80")
//	Dial("tcp", "google.com:http")
//	Dial("tcp", "[2001:db8::1]:http")
//	Dial("tcp", "[fe80::1%lo0]:80")
//
// For IP networks, the network must be "ip", "ip4" or "ip6" followed
// by a colon and a protocol number or name and the addr must be a
// literal IP address.
//
// Examples:
//	Dial("ip4:1", "127.0.0.1")
//	Dial("ip6:ospf", "::1")
//
// For Unix networks, the address must be a file system path.
func Dial(network, address string) (Conn, error) {
	var d Dialer
	return d.Dial(network, address)
}

// DialTimeout acts like Dial but takes a timeout.
// The timeout includes name resolution, if required.
func DialTimeout(network, address string, timeout time.Duration) (Conn, error) {
	d := Dialer{Timeout: timeout}
	return d.Dial(network, address)
}

// Dial connects to the address on the named network.
//
// See func Dial for a description of the network and address
// parameters.
func (d *Dialer) Dial(network, address string) (Conn, error) {
	ra, err := resolveAddr("dial", network, address, d.deadline())
	if err != nil {
		return nil, &OpError{Op: "dial", Net: network, Addr: nil, Err: err}
	}
	var dialer func(deadline time.Time) (Conn, error)
	if ras, ok := ra.(addrList); ok && d.DualStack && network == "tcp" {
		dialer = func(deadline time.Time) (Conn, error) {
			return dialMulti(network, address, d.LocalAddr, ras, deadline)
		}
	} else {
		dialer = func(deadline time.Time) (Conn, error) {
			return dialSingle(network, address, d.LocalAddr, ra.toAddr(), deadline)
		}
	}
	c, err := dial(network, ra.toAddr(), dialer, d.deadline())
	if d.KeepAlive > 0 && err == nil {
		if tc, ok := c.(*TCPConn); ok {
			tc.SetKeepAlive(true)
			tc.SetKeepAlivePeriod(d.KeepAlive)
			testHookSetKeepAlive()
		}
	}
	return c, err
}

var testHookSetKeepAlive = func() {} // changed by dial_test.go

// dialMulti attempts to establish connections to each destination of
// the list of addresses. It will return the first established
// connection and close the other connections. Otherwise it returns
// error on the last attempt.
func dialMulti(net, addr string, la Addr, ras addrList, deadline time.Time) (Conn, error) {
	type racer struct {
		Conn
		error
	}
	// Sig controls the flow of dial results on lane. It passes a
	// token to the next racer and also indicates the end of flow
	// by using closed channel.
	sig := make(chan bool, 1)
	lane := make(chan racer, 1)
	for _, ra := range ras {
		go func(ra Addr) {
			c, err := dialSingle(net, addr, la, ra, deadline)
			if _, ok := <-sig; ok {
				lane <- racer{c, err}
			} else if err == nil {
				// We have to return the resources
				// that belong to the other
				// connections here for avoiding
				// unnecessary resource starvation.
				c.Close()
			}
		}(ra.toAddr())
	}
	defer close(sig)
	lastErr := errTimeout
	nracers := len(ras)
	for nracers > 0 {
		sig <- true
		racer := <-lane
		if racer.error == nil {
			return racer.Conn, nil
		}
		lastErr = racer.error
		nracers--
	}
	return nil, lastErr
}

// dialSingle attempts to establish and returns a single connection to
// the destination address.
func dialSingle(net, addr string, la, ra Addr, deadline time.Time) (c Conn, err error) {
	if la != nil && la.Network() != ra.Network() {
		return nil, &OpError{Op: "dial", Net: net, Addr: ra, Err: errors.New("mismatched local address type " + la.Network())}
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
		return nil, &OpError{Op: "dial", Net: net, Addr: ra, Err: &AddrError{Err: "unexpected address type", Addr: addr}}
	}
	if err != nil {
		return nil, err // c is non-nil interface containing nil pointer
	}
	return c, nil
}

// Listen announces on the local network address laddr.
// The network net must be a stream-oriented network: "tcp", "tcp4",
// "tcp6", "unix" or "unixpacket".
// See Dial for the syntax of laddr.
func Listen(net, laddr string) (Listener, error) {
	la, err := resolveAddr("listen", net, laddr, noDeadline)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: nil, Err: err}
	}
	var l Listener
	switch la := la.toAddr().(type) {
	case *TCPAddr:
		l, err = ListenTCP(net, la)
	case *UnixAddr:
		l, err = ListenUnix(net, la)
	default:
		return nil, &OpError{Op: "listen", Net: net, Addr: la, Err: &AddrError{Err: "unexpected address type", Addr: laddr}}
	}
	if err != nil {
		return nil, err // l is non-nil interface containing nil pointer
	}
	return l, nil
}

// ListenPacket announces on the local network address laddr.
// The network net must be a packet-oriented network: "udp", "udp4",
// "udp6", "ip", "ip4", "ip6" or "unixgram".
// See Dial for the syntax of laddr.
func ListenPacket(net, laddr string) (PacketConn, error) {
	la, err := resolveAddr("listen", net, laddr, noDeadline)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: nil, Err: err}
	}
	var l PacketConn
	switch la := la.toAddr().(type) {
	case *UDPAddr:
		l, err = ListenUDP(net, la)
	case *IPAddr:
		l, err = ListenIP(net, la)
	case *UnixAddr:
		l, err = ListenUnixgram(net, la)
	default:
		return nil, &OpError{Op: "listen", Net: net, Addr: la, Err: &AddrError{Err: "unexpected address type", Addr: laddr}}
	}
	if err != nil {
		return nil, err // l is non-nil interface containing nil pointer
	}
	return l, nil
}

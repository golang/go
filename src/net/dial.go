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
	// When dialing a name with multiple IP addresses, the timeout
	// may be divided between them.
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

	// DualStack enables RFC 6555-compliant "Happy Eyeballs" dialing
	// when the network is "tcp" and the destination is a host name
	// with both IPv4 and IPv6 addresses. This allows a client to
	// tolerate networks where one address family is silently broken.
	DualStack bool

	// FallbackDelay specifies the length of time to wait before
	// spawning a fallback connection, when DualStack is enabled.
	// If zero, a default delay of 300ms is used.
	FallbackDelay time.Duration

	// KeepAlive specifies the keep-alive period for an active
	// network connection.
	// If zero, keep-alives are not enabled. Network protocols
	// that do not support keep-alives ignore this field.
	KeepAlive time.Duration

	// Cancel is an optional channel whose closure indicates that
	// the dial should be canceled. Not all types of dials support
	// cancelation.
	Cancel <-chan struct{}
}

// Return either now+Timeout or Deadline, whichever comes first.
// Or zero, if neither is set.
func (d *Dialer) deadline(now time.Time) time.Time {
	if d.Timeout == 0 {
		return d.Deadline
	}
	timeoutDeadline := now.Add(d.Timeout)
	if d.Deadline.IsZero() || timeoutDeadline.Before(d.Deadline) {
		return timeoutDeadline
	} else {
		return d.Deadline
	}
}

// partialDeadline returns the deadline to use for a single address,
// when multiple addresses are pending.
func partialDeadline(now, deadline time.Time, addrsRemaining int) (time.Time, error) {
	if deadline.IsZero() {
		return deadline, nil
	}
	timeRemaining := deadline.Sub(now)
	if timeRemaining <= 0 {
		return time.Time{}, errTimeout
	}
	// Tentatively allocate equal time to each remaining address.
	timeout := timeRemaining / time.Duration(addrsRemaining)
	// If the time per address is too short, steal from the end of the list.
	const saneMinimum = 2 * time.Second
	if timeout < saneMinimum {
		if timeRemaining < saneMinimum {
			timeout = timeRemaining
		} else {
			timeout = saneMinimum
		}
	}
	return now.Add(timeout), nil
}

func (d *Dialer) fallbackDelay() time.Duration {
	if d.FallbackDelay > 0 {
		return d.FallbackDelay
	} else {
		return 300 * time.Millisecond
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

func resolveAddrList(op, net, addr string, deadline time.Time) (addrList, error) {
	afnet, _, err := parseNetwork(net)
	if err != nil {
		return nil, err
	}
	if op == "dial" && addr == "" {
		return nil, errMissingAddress
	}
	switch afnet {
	case "unix", "unixgram", "unixpacket":
		addr, err := ResolveUnixAddr(afnet, addr)
		if err != nil {
			return nil, err
		}
		return addrList{addr}, nil
	}
	return internetAddrList(afnet, addr, deadline)
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
// If the host is empty, as in ":80", the local system is assumed.
//
// Examples:
//	Dial("tcp", "12.34.56.78:80")
//	Dial("tcp", "google.com:http")
//	Dial("tcp", "[2001:db8::1]:http")
//	Dial("tcp", "[fe80::1%lo0]:80")
//	Dial("tcp", ":80")
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

// dialContext holds common state for all dial operations.
type dialContext struct {
	Dialer
	network, address string
	finalDeadline    time.Time
}

// Dial connects to the address on the named network.
//
// See func Dial for a description of the network and address
// parameters.
func (d *Dialer) Dial(network, address string) (Conn, error) {
	finalDeadline := d.deadline(time.Now())
	addrs, err := resolveAddrList("dial", network, address, finalDeadline)
	if err != nil {
		return nil, &OpError{Op: "dial", Net: network, Source: nil, Addr: nil, Err: err}
	}

	ctx := &dialContext{
		Dialer:        *d,
		network:       network,
		address:       address,
		finalDeadline: finalDeadline,
	}

	var primaries, fallbacks addrList
	if d.DualStack && network == "tcp" {
		primaries, fallbacks = addrs.partition(isIPv4)
	} else {
		primaries = addrs
	}

	var c Conn
	if len(fallbacks) == 0 {
		// dialParallel can accept an empty fallbacks list,
		// but this shortcut avoids the goroutine/channel overhead.
		c, err = dialSerial(ctx, primaries, nil)
	} else {
		c, err = dialParallel(ctx, primaries, fallbacks)
	}

	if d.KeepAlive > 0 && err == nil {
		if tc, ok := c.(*TCPConn); ok {
			setKeepAlive(tc.fd, true)
			setKeepAlivePeriod(tc.fd, d.KeepAlive)
			testHookSetKeepAlive()
		}
	}
	return c, err
}

// dialParallel races two copies of dialSerial, giving the first a
// head start. It returns the first established connection and
// closes the others. Otherwise it returns an error from the first
// primary address.
func dialParallel(ctx *dialContext, primaries, fallbacks addrList) (Conn, error) {
	results := make(chan dialResult) // unbuffered, so dialSerialAsync can detect race loss & cleanup
	cancel := make(chan struct{})
	defer close(cancel)

	// Spawn the primary racer.
	go dialSerialAsync(ctx, primaries, nil, cancel, results)

	// Spawn the fallback racer.
	fallbackTimer := time.NewTimer(ctx.fallbackDelay())
	go dialSerialAsync(ctx, fallbacks, fallbackTimer, cancel, results)

	var primaryErr error
	for nracers := 2; nracers > 0; nracers-- {
		res := <-results
		// If we're still waiting for a connection, then hasten the delay.
		// Otherwise, disable the Timer and let cancel take over.
		if fallbackTimer.Stop() && res.error != nil {
			fallbackTimer.Reset(0)
		}
		if res.error == nil {
			return res.Conn, nil
		}
		if res.primary {
			primaryErr = res.error
		}
	}
	return nil, primaryErr
}

type dialResult struct {
	Conn
	error
	primary bool
}

// dialSerialAsync runs dialSerial after some delay, and returns the
// resulting connection through a channel. When racing two connections,
// the primary goroutine uses a nil timer to omit the delay.
func dialSerialAsync(ctx *dialContext, ras addrList, timer *time.Timer, cancel <-chan struct{}, results chan<- dialResult) {
	if timer != nil {
		// We're in the fallback goroutine; sleep before connecting.
		select {
		case <-timer.C:
		case <-cancel:
			return
		}
	}
	c, err := dialSerial(ctx, ras, cancel)
	select {
	case results <- dialResult{c, err, timer == nil}:
		// We won the race.
	case <-cancel:
		// The other goroutine won the race.
		if c != nil {
			c.Close()
		}
	}
}

// dialSerial connects to a list of addresses in sequence, returning
// either the first successful connection, or the first error.
func dialSerial(ctx *dialContext, ras addrList, cancel <-chan struct{}) (Conn, error) {
	var firstErr error // The error from the first address is most relevant.

	for i, ra := range ras {
		select {
		case <-cancel:
			return nil, &OpError{Op: "dial", Net: ctx.network, Source: ctx.LocalAddr, Addr: ra, Err: errCanceled}
		default:
		}

		partialDeadline, err := partialDeadline(time.Now(), ctx.finalDeadline, len(ras)-i)
		if err != nil {
			// Ran out of time.
			if firstErr == nil {
				firstErr = &OpError{Op: "dial", Net: ctx.network, Source: ctx.LocalAddr, Addr: ra, Err: err}
			}
			break
		}

		// dialTCP does not support cancelation (see golang.org/issue/11225),
		// so if cancel fires, we'll continue trying to connect until the next
		// timeout, or return a spurious connection for the caller to close.
		dialer := func(d time.Time) (Conn, error) {
			return dialSingle(ctx, ra, d)
		}
		c, err := dial(ctx.network, ra, dialer, partialDeadline)
		if err == nil {
			return c, nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}

	if firstErr == nil {
		firstErr = &OpError{Op: "dial", Net: ctx.network, Source: nil, Addr: nil, Err: errMissingAddress}
	}
	return nil, firstErr
}

// dialSingle attempts to establish and returns a single connection to
// the destination address. This must be called through the OS-specific
// dial function, because some OSes don't implement the deadline feature.
func dialSingle(ctx *dialContext, ra Addr, deadline time.Time) (c Conn, err error) {
	la := ctx.LocalAddr
	if la != nil && la.Network() != ra.Network() {
		return nil, &OpError{Op: "dial", Net: ctx.network, Source: la, Addr: ra, Err: errors.New("mismatched local address type " + la.Network())}
	}
	switch ra := ra.(type) {
	case *TCPAddr:
		la, _ := la.(*TCPAddr)
		c, err = testHookDialTCP(ctx.network, la, ra, deadline, ctx.Cancel)
	case *UDPAddr:
		la, _ := la.(*UDPAddr)
		c, err = dialUDP(ctx.network, la, ra, deadline)
	case *IPAddr:
		la, _ := la.(*IPAddr)
		c, err = dialIP(ctx.network, la, ra, deadline)
	case *UnixAddr:
		la, _ := la.(*UnixAddr)
		c, err = dialUnix(ctx.network, la, ra, deadline)
	default:
		return nil, &OpError{Op: "dial", Net: ctx.network, Source: la, Addr: ra, Err: &AddrError{Err: "unexpected address type", Addr: ctx.address}}
	}
	if err != nil {
		return nil, err // c is non-nil interface containing nil pointer
	}
	return c, nil
}

// Listen announces on the local network address laddr.
// The network net must be a stream-oriented network: "tcp", "tcp4",
// "tcp6", "unix" or "unixpacket".
// For TCP and UDP, the syntax of laddr is "host:port", like "127.0.0.1:8080".
// If host is omitted, as in ":8080", Listen listens on all available interfaces
// instead of just the interface with the given host address.
// See Dial for more details about address syntax.
func Listen(net, laddr string) (Listener, error) {
	addrs, err := resolveAddrList("listen", net, laddr, noDeadline)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Source: nil, Addr: nil, Err: err}
	}
	var l Listener
	switch la := addrs.first(isIPv4).(type) {
	case *TCPAddr:
		l, err = ListenTCP(net, la)
	case *UnixAddr:
		l, err = ListenUnix(net, la)
	default:
		return nil, &OpError{Op: "listen", Net: net, Source: nil, Addr: la, Err: &AddrError{Err: "unexpected address type", Addr: laddr}}
	}
	if err != nil {
		return nil, err // l is non-nil interface containing nil pointer
	}
	return l, nil
}

// ListenPacket announces on the local network address laddr.
// The network net must be a packet-oriented network: "udp", "udp4",
// "udp6", "ip", "ip4", "ip6" or "unixgram".
// For TCP and UDP, the syntax of laddr is "host:port", like "127.0.0.1:8080".
// If host is omitted, as in ":8080", ListenPacket listens on all available interfaces
// instead of just the interface with the given host address.
// See Dial for the syntax of laddr.
func ListenPacket(net, laddr string) (PacketConn, error) {
	addrs, err := resolveAddrList("listen", net, laddr, noDeadline)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Source: nil, Addr: nil, Err: err}
	}
	var l PacketConn
	switch la := addrs.first(isIPv4).(type) {
	case *UDPAddr:
		l, err = ListenUDP(net, la)
	case *IPAddr:
		l, err = ListenIP(net, la)
	case *UnixAddr:
		l, err = ListenUnixgram(net, la)
	default:
		return nil, &OpError{Op: "listen", Net: net, Source: nil, Addr: la, Err: &AddrError{Err: "unexpected address type", Addr: laddr}}
	}
	if err != nil {
		return nil, err // l is non-nil interface containing nil pointer
	}
	return l, nil
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

// TODO(rsc):
//	support for raw IP sockets
//	support for raw ethernet sockets

import "os"

// Conn is a generic network connection.
type Conn interface {
	// Read blocks until data is ready from the connection
	// and then reads into b.  It returns the number
	// of bytes read, or 0 if the connection has been closed.
	Read(b []byte) (n int, err os.Error);

	// Write writes the data in b to the connection.
	Write(b []byte) (n int, err os.Error);

	// Close closes the connection.
	Close() os.Error;

	// LocalAddr returns the local network address.
	LocalAddr() string;

	// RemoteAddr returns the remote network address.
	RemoteAddr() string;

	// For packet-based protocols such as UDP,
	// ReadFrom reads the next packet from the network,
	// returning the number of bytes read and the remote
	// address that sent them.
	ReadFrom(b []byte) (n int, addr string, err os.Error);

	// For packet-based protocols such as UDP,
	// WriteTo writes the byte buffer b to the network
	// as a single payload, sending it to the target address.
	WriteTo(addr string, b []byte) (n int, err os.Error);

	// SetReadBuffer sets the size of the operating system's
	// receive buffer associated with the connection.
	SetReadBuffer(bytes int) os.Error;

	// SetReadBuffer sets the size of the operating system's
	// transmit buffer associated with the connection.
	SetWriteBuffer(bytes int) os.Error;

	// SetTimeout sets the read and write deadlines associated
	// with the connection.
	SetTimeout(nsec int64) os.Error;

	// SetReadTimeout sets the time (in nanoseconds) that
	// Read will wait for data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	SetReadTimeout(nsec int64) os.Error;

	// SetWriteTimeout sets the time (in nanoseconds) that
	// Write will wait to send its data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	SetWriteTimeout(nsec int64) os.Error;

	// SetLinger sets the behavior of Close() on a connection
	// which still has data waiting to be sent or to be acknowledged.
	//
	// If sec < 0 (the default), Close returns immediately and
	// the operating system finishes sending the data in the background.
	//
	// If sec == 0, Close returns immediately and the operating system
	// discards any unsent or unacknowledged data.
	//
	// If sec > 0, Close blocks for at most sec seconds waiting for
	// data to be sent and acknowledged.
	SetLinger(sec int) os.Error;

	// SetReuseAddr sets whether it is okay to reuse addresses
	// from recent connections that were not properly closed.
	SetReuseAddr(reuseaddr bool) os.Error;

	// SetDontRoute sets whether outgoing messages should
	// bypass the system routing tables.
	SetDontRoute(dontroute bool) os.Error;

	// SetKeepAlive sets whether the operating system should send
	// keepalive messages on the connection.
	SetKeepAlive(keepalive bool) os.Error;

	// BindToDevice binds a connection to a particular network device.
	BindToDevice(dev string) os.Error;
}

// A Listener is a generic network listener.
// Accept waits for the next connection and Close closes the connection.
type Listener interface {
	Accept() (c Conn, raddr string, err os.Error);
	Close() os.Error;
	Addr() string;	// Listener's network address
}

// Dial connects to the remote address raddr on the network net.
// If the string laddr is not empty, it is used as the local address
// for the connection.
//
// Known networks are "tcp", "tcp4" (IPv4-only), "tcp6" (IPv6-only),
// "udp", "udp4" (IPv4-only), and "udp6" (IPv6-only).
//
// For IP networks, addresses have the form host:port.  If host is
// a literal IPv6 address, it must be enclosed in square brackets.
//
// Examples:
//	Dial("tcp", "", "12.34.56.78:80")
//	Dial("tcp", "", "google.com:80")
//	Dial("tcp", "", "[de:ad:be:ef::ca:fe]:80")
//	Dial("tcp", "127.0.0.1:123", "127.0.0.1:88")
func Dial(net, laddr, raddr string) (c Conn, err os.Error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
		c, err := DialTCP(net, laddr, raddr);
		if err != nil {
			return nil, err
		}
		return c, nil;
	case "udp", "udp4", "upd6":
		c, err := DialUDP(net, laddr, raddr);
		return c, err;
	case "unix", "unix-dgram":
		c, err := DialUnix(net, laddr, raddr);
		return c, err;
	}
	return nil, &OpError{"dial", net, raddr, UnknownNetworkError(net)};
}

// Listen announces on the local network address laddr.
// The network string net must be "tcp", "tcp4", "tcp6",
// "unix", or "unix-dgram".
func Listen(net, laddr string) (l Listener, err os.Error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
		l, err := ListenTCP(net, laddr);
		if err != nil {
			return nil, err;
		}
		return l, nil;
	case "unix", "unix-dgram":
		l, err := ListenUnix(net, laddr);
		if err != nil {
			return nil, err;
		}
		return l, nil;
	// BUG(rsc): Listen should support UDP.
	}
	return nil, UnknownNetworkError(net);
}

var errMissingAddress = os.ErrorString("missing address")

type OpError struct {
	Op string;
	Net string;
	Addr string;
	Error os.Error;
}

func (e *OpError) String() string {
	s := e.Op;
	if e.Net != "" {
		s += " " + e.Net;
	}
	if e.Addr != "" {
		s += " " + e.Addr;
	}
	s += ": " + e.Error.String();
	return s;
}

type AddrError struct {
	Error string;
	Addr string;
}

func (e *AddrError) String() string {
	s := e.Error;
	if e.Addr != "" {
		s += " " + e.Addr;
	}
	return s;
}

type UnknownNetworkError string
func (e UnknownNetworkError) String() string {
	return "unknown network " + string(e);
}


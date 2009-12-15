// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The net package provides a portable interface to Unix
// networks sockets, including TCP/IP, UDP, domain name
// resolution, and Unix domain sockets.
package net

// TODO(rsc):
//	support for raw IP sockets
//	support for raw ethernet sockets

import "os"

// Addr represents a network end point address.
type Addr interface {
	Network() string // name of the network
	String() string  // string form of address
}

// Conn is a generic stream-oriented network connection.
type Conn interface {
	// Read reads data from the connection.
	// Read can be made to time out and return err == os.EAGAIN
	// after a fixed time limit; see SetTimeout and SetReadTimeout.
	Read(b []byte) (n int, err os.Error)

	// Write writes data to the connection.
	// Write can be made to time out and return err == os.EAGAIN
	// after a fixed time limit; see SetTimeout and SetReadTimeout.
	Write(b []byte) (n int, err os.Error)

	// Close closes the connection.
	Close() os.Error

	// LocalAddr returns the local network address.
	LocalAddr() Addr

	// RemoteAddr returns the remote network address.
	RemoteAddr() Addr

	// SetTimeout sets the read and write deadlines associated
	// with the connection.
	SetTimeout(nsec int64) os.Error

	// SetReadTimeout sets the time (in nanoseconds) that
	// Read will wait for data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	SetReadTimeout(nsec int64) os.Error

	// SetWriteTimeout sets the time (in nanoseconds) that
	// Write will wait to send its data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	SetWriteTimeout(nsec int64) os.Error
}

// PacketConn is a generic packet-oriented network connection.
type PacketConn interface {
	// ReadFrom reads a packet from the connection,
	// copying the payload into b.  It returns the number of
	// bytes copied into b and the return address that
	// was on the packet.
	// ReadFrom can be made to time out and return err == os.EAGAIN
	// after a fixed time limit; see SetTimeout and SetReadTimeout.
	ReadFrom(b []byte) (n int, addr Addr, err os.Error)

	// WriteTo writes a packet with payload b to addr.
	// WriteTo can be made to time out and return err == os.EAGAIN
	// after a fixed time limit; see SetTimeout and SetWriteTimeout.
	// On packet-oriented connections, write timeouts are rare.
	WriteTo(b []byte, addr Addr) (n int, err os.Error)

	// Close closes the connection.
	Close() os.Error

	// LocalAddr returns the local network address.
	LocalAddr() Addr

	// SetTimeout sets the read and write deadlines associated
	// with the connection.
	SetTimeout(nsec int64) os.Error

	// SetReadTimeout sets the time (in nanoseconds) that
	// Read will wait for data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	SetReadTimeout(nsec int64) os.Error

	// SetWriteTimeout sets the time (in nanoseconds) that
	// Write will wait to send its data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	SetWriteTimeout(nsec int64) os.Error
}

// A Listener is a generic network listener for stream-oriented protocols.
// Accept waits for the next connection and Close closes the connection.
type Listener interface {
	Accept() (c Conn, err os.Error)
	Close() os.Error
	Addr() Addr // Listener's network address
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
//
func Dial(net, laddr, raddr string) (c Conn, err os.Error) {
	switch net {
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
		return DialTCP(net, la, ra)
	case "udp", "udp4", "upd6":
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
		return DialUDP(net, la, ra)
	case "unix", "unixgram":
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
		return DialUnix(net, la, ra)
	}
	err = UnknownNetworkError(net)
Error:
	return nil, &OpError{"dial", net + " " + raddr, nil, err}
}

// Listen announces on the local network address laddr.
// The network string net must be a stream-oriented
// network: "tcp", "tcp4", "tcp6", or "unix".
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
	case "unix":
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
	}
	return nil, UnknownNetworkError(net)
}

var errMissingAddress = os.ErrorString("missing address")

type OpError struct {
	Op    string
	Net   string
	Addr  Addr
	Error os.Error
}

func (e *OpError) String() string {
	s := e.Op
	if e.Net != "" {
		s += " " + e.Net
	}
	if e.Addr != nil {
		s += " " + e.Addr.String()
	}
	s += ": " + e.Error.String()
	return s
}

type AddrError struct {
	Error string
	Addr  string
}

func (e *AddrError) String() string {
	s := e.Error
	if e.Addr != "" {
		s += " " + e.Addr
	}
	return s
}

type UnknownNetworkError string

func (e UnknownNetworkError) String() string { return "unknown network " + string(e) }

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The net package provides a portable interface to Unix
// networks sockets, including TCP/IP, UDP, domain name
// resolution, and Unix domain sockets.
package net

// TODO(rsc):
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
	// Read can be made to time out and return a net.Error with Timeout() == true
	// after a fixed time limit; see SetTimeout and SetReadTimeout.
	Read(b []byte) (n int, err os.Error)

	// Write writes data to the connection.
	// Write can be made to time out and return a net.Error with Timeout() == true
	// after a fixed time limit; see SetTimeout and SetWriteTimeout.
	Write(b []byte) (n int, err os.Error)

	// Close closes the connection.
	// The error returned is an os.Error to satisfy io.Closer;
	Close() os.Error

	// LocalAddr returns the local network address.
	LocalAddr() Addr

	// RemoteAddr returns the remote network address.
	RemoteAddr() Addr

	// SetTimeout sets the read and write deadlines associated
	// with the connection.
	SetTimeout(nsec int64) os.Error

	// SetReadTimeout sets the time (in nanoseconds) that
	// Read will wait for data before returning an error with Timeout() == true.
	// Setting nsec == 0 (the default) disables the deadline.
	SetReadTimeout(nsec int64) os.Error

	// SetWriteTimeout sets the time (in nanoseconds) that
	// Write will wait to send its data before returning an error with Timeout() == true.
	// Setting nsec == 0 (the default) disables the deadline.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	SetWriteTimeout(nsec int64) os.Error
}

// An Error represents a network error.
type Error interface {
	os.Error
	Timeout() bool   // Is the error a timeout?
	Temporary() bool // Is the error temporary?
}

// PacketConn is a generic packet-oriented network connection.
type PacketConn interface {
	// ReadFrom reads a packet from the connection,
	// copying the payload into b.  It returns the number of
	// bytes copied into b and the return address that
	// was on the packet.
	// ReadFrom can be made to time out and return
	// an error with Timeout() == true after a fixed time limit;
	// see SetTimeout and SetReadTimeout.
	ReadFrom(b []byte) (n int, addr Addr, err os.Error)

	// WriteTo writes a packet with payload b to addr.
	// WriteTo can be made to time out and return
	// an error with Timeout() == true after a fixed time limit;
	// see SetTimeout and SetWriteTimeout.
	// On packet-oriented connections, write timeouts are rare.
	WriteTo(b []byte, addr Addr) (n int, err os.Error)

	// Close closes the connection.
	// The error returned is an os.Error to satisfy io.Closer;
	Close() os.Error

	// LocalAddr returns the local network address.
	LocalAddr() Addr

	// SetTimeout sets the read and write deadlines associated
	// with the connection.
	SetTimeout(nsec int64) os.Error

	// SetReadTimeout sets the time (in nanoseconds) that
	// Read will wait for data before returning an error with Timeout() == true.
	// Setting nsec == 0 (the default) disables the deadline.
	SetReadTimeout(nsec int64) os.Error

	// SetWriteTimeout sets the time (in nanoseconds) that
	// Write will wait to send its data before returning an error with Timeout() == true.
	// Setting nsec == 0 (the default) disables the deadline.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	SetWriteTimeout(nsec int64) os.Error
}

// A Listener is a generic network listener for stream-oriented protocols.
type Listener interface {
	// Accept waits for and returns the next connection to the listener.
	Accept() (c Conn, err os.Error)

	// Close closes the listener.
	// The error returned is an os.Error to satisfy io.Closer;
	Close() os.Error

	// Addr returns the listener's network address.
	Addr() Addr
}

var errMissingAddress = os.ErrorString("missing address")

type OpError struct {
	Op    string
	Net   string
	Addr  Addr
	Error os.Error
}

func (e *OpError) String() string {
	if e == nil {
		return "<nil>"
	}
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

type temporary interface {
	Temporary() bool
}

func (e *OpError) Temporary() bool {
	t, ok := e.Error.(temporary)
	return ok && t.Temporary()
}

type timeout interface {
	Timeout() bool
}

func (e *OpError) Timeout() bool {
	t, ok := e.Error.(timeout)
	return ok && t.Timeout()
}

type AddrError struct {
	Error string
	Addr  string
}

func (e *AddrError) String() string {
	if e == nil {
		return "<nil>"
	}
	s := e.Error
	if e.Addr != "" {
		s += " " + e.Addr
	}
	return s
}

func (e *AddrError) Temporary() bool {
	return false
}

func (e *AddrError) Timeout() bool {
	return false
}

type UnknownNetworkError string

func (e UnknownNetworkError) String() string  { return "unknown network " + string(e) }
func (e UnknownNetworkError) Temporary() bool { return false }
func (e UnknownNetworkError) Timeout() bool   { return false }

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package net provides a portable interface for network I/O, including
TCP/IP, UDP, domain name resolution, and Unix domain sockets.

Although the package provides access to low-level networking
primitives, most clients will need only the basic interface provided
by the Dial, Listen, and Accept functions and the associated
Conn and Listener interfaces. The crypto/tls package uses
the same interfaces and similar Dial and Listen functions.

The Dial function connects to a server:

	conn, err := net.Dial("tcp", "google.com:80")
	if err != nil {
		// handle error
	}
	fmt.Fprintf(conn, "GET / HTTP/1.0\r\n\r\n")
	status, err := bufio.NewReader(conn).ReadString('\n')
	// ...

The Listen function creates servers:

	ln, err := net.Listen("tcp", ":8080")
	if err != nil {
		// handle error
	}
	for {
		conn, err := ln.Accept()
		if err != nil {
			// handle error
		}
		go handleConnection(conn)
	}
*/
package net

// TODO(rsc):
//	support for raw ethernet sockets

import (
	"errors"
	"io"
	"os"
	"syscall"
	"time"
)

// Addr represents a network end point address.
type Addr interface {
	Network() string // name of the network
	String() string  // string form of address
}

// Conn is a generic stream-oriented network connection.
//
// Multiple goroutines may invoke methods on a Conn simultaneously.
type Conn interface {
	// Read reads data from the connection.
	// Read can be made to time out and return a Error with Timeout() == true
	// after a fixed time limit; see SetDeadline and SetReadDeadline.
	Read(b []byte) (n int, err error)

	// Write writes data to the connection.
	// Write can be made to time out and return a Error with Timeout() == true
	// after a fixed time limit; see SetDeadline and SetWriteDeadline.
	Write(b []byte) (n int, err error)

	// Close closes the connection.
	// Any blocked Read or Write operations will be unblocked and return errors.
	Close() error

	// LocalAddr returns the local network address.
	LocalAddr() Addr

	// RemoteAddr returns the remote network address.
	RemoteAddr() Addr

	// SetDeadline sets the read and write deadlines associated
	// with the connection. It is equivalent to calling both
	// SetReadDeadline and SetWriteDeadline.
	//
	// A deadline is an absolute time after which I/O operations
	// fail with a timeout (see type Error) instead of
	// blocking. The deadline applies to all future I/O, not just
	// the immediately following call to Read or Write.
	//
	// An idle timeout can be implemented by repeatedly extending
	// the deadline after successful Read or Write calls.
	//
	// A zero value for t means I/O operations will not time out.
	SetDeadline(t time.Time) error

	// SetReadDeadline sets the deadline for future Read calls.
	// A zero value for t means Read will not time out.
	SetReadDeadline(t time.Time) error

	// SetWriteDeadline sets the deadline for future Write calls.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	// A zero value for t means Write will not time out.
	SetWriteDeadline(t time.Time) error
}

type conn struct {
	fd *netFD
}

func (c *conn) ok() bool { return c != nil && c.fd != nil }

// Implementation of the Conn interface.

// Read implements the Conn Read method.
func (c *conn) Read(b []byte) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	return c.fd.Read(b)
}

// Write implements the Conn Write method.
func (c *conn) Write(b []byte) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	return c.fd.Write(b)
}

// Close closes the connection.
func (c *conn) Close() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.Close()
}

// LocalAddr returns the local network address.
func (c *conn) LocalAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.laddr
}

// RemoteAddr returns the remote network address.
func (c *conn) RemoteAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.raddr
}

// SetDeadline implements the Conn SetDeadline method.
func (c *conn) SetDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.setDeadline(t)
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *conn) SetReadDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.setReadDeadline(t)
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *conn) SetWriteDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.setWriteDeadline(t)
}

// SetReadBuffer sets the size of the operating system's
// receive buffer associated with the connection.
func (c *conn) SetReadBuffer(bytes int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setReadBuffer(c.fd, bytes)
}

// SetWriteBuffer sets the size of the operating system's
// transmit buffer associated with the connection.
func (c *conn) SetWriteBuffer(bytes int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setWriteBuffer(c.fd, bytes)
}

// File sets the underlying os.File to blocking mode and returns a copy.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
//
// The returned os.File's file descriptor is different from the connection's.
// Attempting to change properties of the original using this duplicate
// may or may not have the desired effect.
func (c *conn) File() (f *os.File, err error) { return c.fd.dup() }

// An Error represents a network error.
type Error interface {
	error
	Timeout() bool   // Is the error a timeout?
	Temporary() bool // Is the error temporary?
}

// PacketConn is a generic packet-oriented network connection.
//
// Multiple goroutines may invoke methods on a PacketConn simultaneously.
type PacketConn interface {
	// ReadFrom reads a packet from the connection,
	// copying the payload into b.  It returns the number of
	// bytes copied into b and the return address that
	// was on the packet.
	// ReadFrom can be made to time out and return
	// an error with Timeout() == true after a fixed time limit;
	// see SetDeadline and SetReadDeadline.
	ReadFrom(b []byte) (n int, addr Addr, err error)

	// WriteTo writes a packet with payload b to addr.
	// WriteTo can be made to time out and return
	// an error with Timeout() == true after a fixed time limit;
	// see SetDeadline and SetWriteDeadline.
	// On packet-oriented connections, write timeouts are rare.
	WriteTo(b []byte, addr Addr) (n int, err error)

	// Close closes the connection.
	// Any blocked ReadFrom or WriteTo operations will be unblocked and return errors.
	Close() error

	// LocalAddr returns the local network address.
	LocalAddr() Addr

	// SetDeadline sets the read and write deadlines associated
	// with the connection.
	SetDeadline(t time.Time) error

	// SetReadDeadline sets the deadline for future Read calls.
	// If the deadline is reached, Read will fail with a timeout
	// (see type Error) instead of blocking.
	// A zero value for t means Read will not time out.
	SetReadDeadline(t time.Time) error

	// SetWriteDeadline sets the deadline for future Write calls.
	// If the deadline is reached, Write will fail with a timeout
	// (see type Error) instead of blocking.
	// A zero value for t means Write will not time out.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	SetWriteDeadline(t time.Time) error
}

var listenerBacklog = maxListenerBacklog()

// A Listener is a generic network listener for stream-oriented protocols.
//
// Multiple goroutines may invoke methods on a Listener simultaneously.
type Listener interface {
	// Accept waits for and returns the next connection to the listener.
	Accept() (c Conn, err error)

	// Close closes the listener.
	// Any blocked Accept operations will be unblocked and return errors.
	Close() error

	// Addr returns the listener's network address.
	Addr() Addr
}

// Various errors contained in OpError.
var (
	// For connection setup and write operations.
	errMissingAddress = errors.New("missing address")

	// For both read and write operations.
	errTimeout          error = &timeoutError{}
	errClosing                = errors.New("use of closed network connection")
	ErrWriteToConnected       = errors.New("use of WriteTo with pre-connected connection")
)

// OpError is the error type usually returned by functions in the net
// package. It describes the operation, network type, and address of
// an error.
type OpError struct {
	// Op is the operation which caused the error, such as
	// "read" or "write".
	Op string

	// Net is the network type on which this error occurred,
	// such as "tcp" or "udp6".
	Net string

	// Addr is the network address on which this error occurred.
	Addr Addr

	// Err is the error that occurred during the operation.
	Err error
}

func (e *OpError) Error() string {
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
	s += ": " + e.Err.Error()
	return s
}

type temporary interface {
	Temporary() bool
}

func (e *OpError) Temporary() bool {
	t, ok := e.Err.(temporary)
	return ok && t.Temporary()
}

var noDeadline = time.Time{}

type timeout interface {
	Timeout() bool
}

func (e *OpError) Timeout() bool {
	t, ok := e.Err.(timeout)
	return ok && t.Timeout()
}

type timeoutError struct{}

func (e *timeoutError) Error() string   { return "i/o timeout" }
func (e *timeoutError) Timeout() bool   { return true }
func (e *timeoutError) Temporary() bool { return true }

type AddrError struct {
	Err  string
	Addr string
}

func (e *AddrError) Error() string {
	if e == nil {
		return "<nil>"
	}
	s := e.Err
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

func (e UnknownNetworkError) Error() string   { return "unknown network " + string(e) }
func (e UnknownNetworkError) Temporary() bool { return false }
func (e UnknownNetworkError) Timeout() bool   { return false }

type InvalidAddrError string

func (e InvalidAddrError) Error() string   { return string(e) }
func (e InvalidAddrError) Timeout() bool   { return false }
func (e InvalidAddrError) Temporary() bool { return false }

// DNSConfigError represents an error reading the machine's DNS configuration.
type DNSConfigError struct {
	Err error
}

func (e *DNSConfigError) Error() string {
	return "error reading DNS config: " + e.Err.Error()
}

func (e *DNSConfigError) Timeout() bool   { return false }
func (e *DNSConfigError) Temporary() bool { return false }

type writerOnly struct {
	io.Writer
}

// Fallback implementation of io.ReaderFrom's ReadFrom, when sendfile isn't
// applicable.
func genericReadFrom(w io.Writer, r io.Reader) (n int64, err error) {
	// Use wrapper to hide existing r.ReadFrom from io.Copy.
	return io.Copy(writerOnly{w}, r)
}

// Limit the number of concurrent cgo-using goroutines, because
// each will block an entire operating system thread. The usual culprit
// is resolving many DNS names in separate goroutines but the DNS
// server is not responding. Then the many lookups each use a different
// thread, and the system or the program runs out of threads.

var threadLimit = make(chan struct{}, 500)

// Using send for acquire is fine here because we are not using this
// to protect any memory. All we care about is the number of goroutines
// making calls at a time.

func acquireThread() {
	threadLimit <- struct{}{}
}

func releaseThread() {
	<-threadLimit
}

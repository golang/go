// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"os"
	"sync"
	"syscall"
	"time"
)

// BUG(mikio): On JS, WASIP1 and Plan 9, methods and functions related
// to UnixConn and UnixListener are not implemented.

// BUG(mikio): On Windows, methods and functions related to UnixConn
// and UnixListener don't work for "unixgram" and "unixpacket".

// UnixAddr represents the address of a Unix domain socket end point.
type UnixAddr struct {
	Name string
	Net  string
}

// Network returns the address's network name, "unix", "unixgram" or
// "unixpacket".
func (a *UnixAddr) Network() string {
	return a.Net
}

func (a *UnixAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	return a.Name
}

func (a *UnixAddr) isWildcard() bool {
	return a == nil || a.Name == ""
}

func (a *UnixAddr) opAddr() Addr {
	if a == nil {
		return nil
	}
	return a
}

// ResolveUnixAddr returns an address of Unix domain socket end point.
//
// The network must be a Unix network name.
//
// See func Dial for a description of the network and address
// parameters.
func ResolveUnixAddr(network, address string) (*UnixAddr, error) {
	switch network {
	case "unix", "unixgram", "unixpacket":
		return &UnixAddr{Name: address, Net: network}, nil
	default:
		return nil, UnknownNetworkError(network)
	}
}

// UnixConn is an implementation of the Conn interface for connections
// to Unix domain sockets.
type UnixConn struct {
	conn
}

// SyscallConn returns a raw network connection.
// This implements the syscall.Conn interface.
func (c *UnixConn) SyscallConn() (syscall.RawConn, error) {
	if !c.ok() {
		return nil, syscall.EINVAL
	}
	return newRawConn(c.fd), nil
}

// CloseRead shuts down the reading side of the Unix domain connection.
// Most callers should just use Close.
func (c *UnixConn) CloseRead() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := c.fd.closeRead(); err != nil {
		return &OpError{Op: "close", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// CloseWrite shuts down the writing side of the Unix domain connection.
// Most callers should just use Close.
func (c *UnixConn) CloseWrite() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	if err := c.fd.closeWrite(); err != nil {
		return &OpError{Op: "close", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return nil
}

// ReadFromUnix acts like ReadFrom but returns a UnixAddr.
func (c *UnixConn) ReadFromUnix(b []byte) (int, *UnixAddr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, addr, err := c.readFrom(b)
	if err != nil {
		err = &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return n, addr, err
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UnixConn) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, addr, err := c.readFrom(b)
	if err != nil {
		err = &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	if addr == nil {
		return n, nil, err
	}
	return n, addr, err
}

// ReadMsgUnix reads a message from c, copying the payload into b and
// the associated out-of-band data into oob. It returns the number of
// bytes copied into b, the number of bytes copied into oob, the flags
// that were set on the message and the source address of the message.
//
// Note that if len(b) == 0 and len(oob) > 0, this function will still
// read (and discard) 1 byte from the connection.
func (c *UnixConn) ReadMsgUnix(b, oob []byte) (n, oobn, flags int, addr *UnixAddr, err error) {
	if !c.ok() {
		return 0, 0, 0, nil, syscall.EINVAL
	}
	n, oobn, flags, addr, err = c.readMsg(b, oob)
	if err != nil {
		err = &OpError{Op: "read", Net: c.fd.net, Source: c.fd.laddr, Addr: c.fd.raddr, Err: err}
	}
	return
}

// WriteToUnix acts like WriteTo but takes a UnixAddr.
func (c *UnixConn) WriteToUnix(b []byte, addr *UnixAddr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	n, err := c.writeTo(b, addr)
	if err != nil {
		err = &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: err}
	}
	return n, err
}

// WriteTo implements the PacketConn WriteTo method.
func (c *UnixConn) WriteTo(b []byte, addr Addr) (int, error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	a, ok := addr.(*UnixAddr)
	if !ok {
		return 0, &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr, Err: syscall.EINVAL}
	}
	n, err := c.writeTo(b, a)
	if err != nil {
		err = &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: a.opAddr(), Err: err}
	}
	return n, err
}

// WriteMsgUnix writes a message to addr via c, copying the payload
// from b and the associated out-of-band data from oob. It returns the
// number of payload and out-of-band bytes written.
//
// Note that if len(b) == 0 and len(oob) > 0, this function will still
// write 1 byte to the connection.
func (c *UnixConn) WriteMsgUnix(b, oob []byte, addr *UnixAddr) (n, oobn int, err error) {
	if !c.ok() {
		return 0, 0, syscall.EINVAL
	}
	n, oobn, err = c.writeMsg(b, oob, addr)
	if err != nil {
		err = &OpError{Op: "write", Net: c.fd.net, Source: c.fd.laddr, Addr: addr.opAddr(), Err: err}
	}
	return
}

func newUnixConn(fd *netFD) *UnixConn { return &UnixConn{conn{fd}} }

// DialUnix acts like Dial for Unix networks.
//
// The network must be a Unix network name; see func Dial for details.
//
// If laddr is non-nil, it is used as the local address for the
// connection.
func DialUnix(network string, laddr, raddr *UnixAddr) (*UnixConn, error) {
	switch network {
	case "unix", "unixgram", "unixpacket":
	default:
		return nil, &OpError{Op: "dial", Net: network, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: UnknownNetworkError(network)}
	}
	sd := &sysDialer{network: network, address: raddr.String()}
	c, err := sd.dialUnix(context.Background(), laddr, raddr)
	if err != nil {
		return nil, &OpError{Op: "dial", Net: network, Source: laddr.opAddr(), Addr: raddr.opAddr(), Err: err}
	}
	return c, nil
}

// UnixListener is a Unix domain socket listener. Clients should
// typically use variables of type Listener instead of assuming Unix
// domain sockets.
type UnixListener struct {
	fd         *netFD
	path       string
	unlink     bool
	unlinkOnce sync.Once
}

func (ln *UnixListener) ok() bool { return ln != nil && ln.fd != nil }

// SyscallConn returns a raw network connection.
// This implements the syscall.Conn interface.
//
// The returned RawConn only supports calling Control. Read and
// Write return an error.
func (l *UnixListener) SyscallConn() (syscall.RawConn, error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	return newRawListener(l.fd), nil
}

// AcceptUnix accepts the next incoming call and returns the new
// connection.
func (l *UnixListener) AcceptUnix() (*UnixConn, error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	c, err := l.accept()
	if err != nil {
		return nil, &OpError{Op: "accept", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return c, nil
}

// Accept implements the Accept method in the Listener interface.
// Returned connections will be of type *UnixConn.
func (l *UnixListener) Accept() (Conn, error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	c, err := l.accept()
	if err != nil {
		return nil, &OpError{Op: "accept", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return c, nil
}

// Close stops listening on the Unix address. Already accepted
// connections are not closed.
func (l *UnixListener) Close() error {
	if !l.ok() {
		return syscall.EINVAL
	}
	if err := l.close(); err != nil {
		return &OpError{Op: "close", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return nil
}

// Addr returns the listener's network address.
// The Addr returned is shared by all invocations of Addr, so
// do not modify it.
func (l *UnixListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *UnixListener) SetDeadline(t time.Time) error {
	if !l.ok() {
		return syscall.EINVAL
	}
	return l.fd.SetDeadline(t)
}

// File returns a copy of the underlying os.File.
// It is the caller's responsibility to close f when finished.
// Closing l does not affect f, and closing f does not affect l.
//
// The returned os.File's file descriptor is different from the
// connection's. Attempting to change properties of the original
// using this duplicate may or may not have the desired effect.
func (l *UnixListener) File() (f *os.File, err error) {
	if !l.ok() {
		return nil, syscall.EINVAL
	}
	f, err = l.file()
	if err != nil {
		err = &OpError{Op: "file", Net: l.fd.net, Source: nil, Addr: l.fd.laddr, Err: err}
	}
	return
}

// ListenUnix acts like Listen for Unix networks.
//
// The network must be "unix" or "unixpacket".
func ListenUnix(network string, laddr *UnixAddr) (*UnixListener, error) {
	switch network {
	case "unix", "unixpacket":
	default:
		return nil, &OpError{Op: "listen", Net: network, Source: nil, Addr: laddr.opAddr(), Err: UnknownNetworkError(network)}
	}
	if laddr == nil {
		return nil, &OpError{Op: "listen", Net: network, Source: nil, Addr: laddr.opAddr(), Err: errMissingAddress}
	}
	sl := &sysListener{network: network, address: laddr.String()}
	ln, err := sl.listenUnix(context.Background(), laddr)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: network, Source: nil, Addr: laddr.opAddr(), Err: err}
	}
	return ln, nil
}

// ListenUnixgram acts like ListenPacket for Unix networks.
//
// The network must be "unixgram".
func ListenUnixgram(network string, laddr *UnixAddr) (*UnixConn, error) {
	switch network {
	case "unixgram":
	default:
		return nil, &OpError{Op: "listen", Net: network, Source: nil, Addr: laddr.opAddr(), Err: UnknownNetworkError(network)}
	}
	if laddr == nil {
		return nil, &OpError{Op: "listen", Net: network, Source: nil, Addr: nil, Err: errMissingAddress}
	}
	sl := &sysListener{network: network, address: laddr.String()}
	c, err := sl.listenUnixgram(context.Background(), laddr)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: network, Source: nil, Addr: laddr.opAddr(), Err: err}
	}
	return c, nil
}

// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd linux netbsd openbsd windows

package net

import (
	"errors"
	"os"
	"syscall"
	"time"
)

func unixSocket(net string, laddr, raddr sockaddr, mode string, deadline time.Time) (*netFD, error) {
	var sotype int
	switch net {
	case "unix":
		sotype = syscall.SOCK_STREAM
	case "unixgram":
		sotype = syscall.SOCK_DGRAM
	case "unixpacket":
		sotype = syscall.SOCK_SEQPACKET
	default:
		return nil, UnknownNetworkError(net)
	}

	switch mode {
	case "dial":
		if laddr != nil && laddr.isWildcard() {
			laddr = nil
		}
		if raddr != nil && raddr.isWildcard() {
			raddr = nil
		}
		if raddr == nil && (sotype != syscall.SOCK_DGRAM || laddr == nil) {
			return nil, errMissingAddress
		}
	case "listen":
	default:
		return nil, errors.New("unknown mode: " + mode)
	}

	f := sockaddrToUnix
	if sotype == syscall.SOCK_DGRAM {
		f = sockaddrToUnixgram
	} else if sotype == syscall.SOCK_SEQPACKET {
		f = sockaddrToUnixpacket
	}

	fd, err := socket(net, syscall.AF_UNIX, sotype, 0, false, laddr, raddr, deadline, f)
	if err != nil {
		return nil, err
	}
	return fd, nil
}

func sockaddrToUnix(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{Name: s.Name, Net: "unix"}
	}
	return nil
}

func sockaddrToUnixgram(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{Name: s.Name, Net: "unixgram"}
	}
	return nil
}

func sockaddrToUnixpacket(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{Name: s.Name, Net: "unixpacket"}
	}
	return nil
}

func sotypeToNet(sotype int) string {
	switch sotype {
	case syscall.SOCK_STREAM:
		return "unix"
	case syscall.SOCK_DGRAM:
		return "unixgram"
	case syscall.SOCK_SEQPACKET:
		return "unixpacket"
	default:
		panic("sotypeToNet unknown socket type")
	}
}

func (a *UnixAddr) family() int {
	return syscall.AF_UNIX
}

func (a *UnixAddr) isWildcard() bool {
	return a == nil || a.Name == ""
}

func (a *UnixAddr) sockaddr(family int) (syscall.Sockaddr, error) {
	if a == nil {
		return nil, nil
	}
	return &syscall.SockaddrUnix{Name: a.Name}, nil
}

// UnixConn is an implementation of the Conn interface for connections
// to Unix domain sockets.
type UnixConn struct {
	conn
}

func newUnixConn(fd *netFD) *UnixConn { return &UnixConn{conn{fd}} }

// ReadFromUnix reads a packet from c, copying the payload into b.  It
// returns the number of bytes copied into b and the source address of
// the packet.
//
// ReadFromUnix can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetReadDeadline.
func (c *UnixConn) ReadFromUnix(b []byte) (n int, addr *UnixAddr, err error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, sa, err := c.fd.ReadFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrUnix:
		if sa.Name != "" {
			addr = &UnixAddr{Name: sa.Name, Net: sotypeToNet(c.fd.sotype)}
		}
	}
	return
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UnixConn) ReadFrom(b []byte) (int, Addr, error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, addr, err := c.ReadFromUnix(b)
	return n, addr.toAddr(), err
}

// ReadMsgUnix reads a packet from c, copying the payload into b and
// the associated out-of-band data into oob.  It returns the number of
// bytes copied into b, the number of bytes copied into oob, the flags
// that were set on the packet, and the source address of the packet.
func (c *UnixConn) ReadMsgUnix(b, oob []byte) (n, oobn, flags int, addr *UnixAddr, err error) {
	if !c.ok() {
		return 0, 0, 0, nil, syscall.EINVAL
	}
	n, oobn, flags, sa, err := c.fd.ReadMsg(b, oob)
	switch sa := sa.(type) {
	case *syscall.SockaddrUnix:
		if sa.Name != "" {
			addr = &UnixAddr{Name: sa.Name, Net: sotypeToNet(c.fd.sotype)}
		}
	}
	return
}

// WriteToUnix writes a packet to addr via c, copying the payload from b.
//
// WriteToUnix can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetDeadline and
// SetWriteDeadline.  On packet-oriented connections, write timeouts
// are rare.
func (c *UnixConn) WriteToUnix(b []byte, addr *UnixAddr) (n int, err error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	if addr == nil {
		return 0, &OpError{Op: "write", Net: c.fd.net, Addr: nil, Err: errMissingAddress}
	}
	if addr.Net != sotypeToNet(c.fd.sotype) {
		return 0, syscall.EAFNOSUPPORT
	}
	sa := &syscall.SockaddrUnix{Name: addr.Name}
	return c.fd.WriteTo(b, sa)
}

// WriteTo implements the PacketConn WriteTo method.
func (c *UnixConn) WriteTo(b []byte, addr Addr) (n int, err error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	a, ok := addr.(*UnixAddr)
	if !ok {
		return 0, &OpError{"write", c.fd.net, addr, syscall.EINVAL}
	}
	return c.WriteToUnix(b, a)
}

// WriteMsgUnix writes a packet to addr via c, copying the payload
// from b and the associated out-of-band data from oob.  It returns
// the number of payload and out-of-band bytes written.
func (c *UnixConn) WriteMsgUnix(b, oob []byte, addr *UnixAddr) (n, oobn int, err error) {
	if !c.ok() {
		return 0, 0, syscall.EINVAL
	}
	if addr != nil {
		if addr.Net != sotypeToNet(c.fd.sotype) {
			return 0, 0, syscall.EAFNOSUPPORT
		}
		sa := &syscall.SockaddrUnix{Name: addr.Name}
		return c.fd.WriteMsg(b, oob, sa)
	}
	return c.fd.WriteMsg(b, oob, nil)
}

// CloseRead shuts down the reading side of the Unix domain connection.
// Most callers should just use Close.
func (c *UnixConn) CloseRead() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.CloseRead()
}

// CloseWrite shuts down the writing side of the Unix domain connection.
// Most callers should just use Close.
func (c *UnixConn) CloseWrite() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return c.fd.CloseWrite()
}

// DialUnix connects to the remote address raddr on the network net,
// which must be "unix", "unixgram" or "unixpacket".  If laddr is not
// nil, it is used as the local address for the connection.
func DialUnix(net string, laddr, raddr *UnixAddr) (*UnixConn, error) {
	switch net {
	case "unix", "unixgram", "unixpacket":
	default:
		return nil, &OpError{Op: "dial", Net: net, Addr: raddr, Err: UnknownNetworkError(net)}
	}
	return dialUnix(net, laddr, raddr, noDeadline)
}

func dialUnix(net string, laddr, raddr *UnixAddr, deadline time.Time) (*UnixConn, error) {
	fd, err := unixSocket(net, laddr, raddr, "dial", deadline)
	if err != nil {
		return nil, &OpError{Op: "dial", Net: net, Addr: raddr, Err: err}
	}
	return newUnixConn(fd), nil
}

// UnixListener is a Unix domain socket listener.  Clients should
// typically use variables of type Listener instead of assuming Unix
// domain sockets.
type UnixListener struct {
	fd   *netFD
	path string
}

// ListenUnix announces on the Unix domain socket laddr and returns a
// Unix listener.  The network net must be "unix" or "unixpacket".
func ListenUnix(net string, laddr *UnixAddr) (*UnixListener, error) {
	switch net {
	case "unix", "unixpacket":
	default:
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: UnknownNetworkError(net)}
	}
	if laddr == nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: nil, Err: errMissingAddress}
	}
	fd, err := unixSocket(net, laddr, nil, "listen", noDeadline)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: err}
	}
	return &UnixListener{fd, fd.laddr.String()}, nil
}

// AcceptUnix accepts the next incoming call and returns the new
// connection.
func (l *UnixListener) AcceptUnix() (*UnixConn, error) {
	if l == nil || l.fd == nil {
		return nil, syscall.EINVAL
	}
	toAddr := sockaddrToUnix
	if l.fd.sotype == syscall.SOCK_SEQPACKET {
		toAddr = sockaddrToUnixpacket
	}
	fd, err := l.fd.accept(toAddr)
	if err != nil {
		return nil, err
	}
	c := newUnixConn(fd)
	return c, nil
}

// Accept implements the Accept method in the Listener interface; it
// waits for the next call and returns a generic Conn.
func (l *UnixListener) Accept() (c Conn, err error) {
	c1, err := l.AcceptUnix()
	if err != nil {
		return nil, err
	}
	return c1, nil
}

// Close stops listening on the Unix address.  Already accepted
// connections are not closed.
func (l *UnixListener) Close() error {
	if l == nil || l.fd == nil {
		return syscall.EINVAL
	}

	// The operating system doesn't clean up
	// the file that announcing created, so
	// we have to clean it up ourselves.
	// There's a race here--we can't know for
	// sure whether someone else has come along
	// and replaced our socket name already--
	// but this sequence (remove then close)
	// is at least compatible with the auto-remove
	// sequence in ListenUnix.  It's only non-Go
	// programs that can mess us up.
	if l.path[0] != '@' {
		syscall.Unlink(l.path)
	}
	return l.fd.Close()
}

// Addr returns the listener's network address.
func (l *UnixListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *UnixListener) SetDeadline(t time.Time) (err error) {
	if l == nil || l.fd == nil {
		return syscall.EINVAL
	}
	return l.fd.setDeadline(t)
}

// File returns a copy of the underlying os.File, set to blocking
// mode.  It is the caller's responsibility to close f when finished.
// Closing l does not affect f, and closing f does not affect l.
//
// The returned os.File's file descriptor is different from the
// connection's.  Attempting to change properties of the original
// using this duplicate may or may not have the desired effect.
func (l *UnixListener) File() (f *os.File, err error) { return l.fd.dup() }

// ListenUnixgram listens for incoming Unix datagram packets addressed
// to the local address laddr.  The network net must be "unixgram".
// The returned connection's ReadFrom and WriteTo methods can be used
// to receive and send packets with per-packet addressing.
func ListenUnixgram(net string, laddr *UnixAddr) (*UnixConn, error) {
	switch net {
	case "unixgram":
	default:
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: UnknownNetworkError(net)}
	}
	if laddr == nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: nil, Err: errMissingAddress}
	}
	fd, err := unixSocket(net, laddr, nil, "listen", noDeadline)
	if err != nil {
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: err}
	}
	return newUnixConn(fd), nil
}

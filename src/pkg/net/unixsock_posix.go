// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin freebsd linux netbsd openbsd windows

// Unix domain sockets

package net

import (
	"os"
	"syscall"
	"time"
)

func unixSocket(net string, laddr, raddr *UnixAddr, mode string) (fd *netFD, err error) {
	var sotype int
	switch net {
	default:
		return nil, UnknownNetworkError(net)
	case "unix":
		sotype = syscall.SOCK_STREAM
	case "unixgram":
		sotype = syscall.SOCK_DGRAM
	case "unixpacket":
		sotype = syscall.SOCK_SEQPACKET
	}

	var la, ra syscall.Sockaddr
	switch mode {
	default:
		panic("unixSocket mode " + mode)

	case "dial":
		if laddr != nil {
			la = &syscall.SockaddrUnix{Name: laddr.Name}
		}
		if raddr != nil {
			ra = &syscall.SockaddrUnix{Name: raddr.Name}
		} else if sotype != syscall.SOCK_DGRAM || laddr == nil {
			return nil, &OpError{Op: mode, Net: net, Err: errMissingAddress}
		}

	case "listen":
		if laddr == nil {
			return nil, &OpError{mode, net, nil, errMissingAddress}
		}
		la = &syscall.SockaddrUnix{Name: laddr.Name}
		if raddr != nil {
			return nil, &OpError{Op: mode, Net: net, Addr: raddr, Err: &AddrError{Err: "unexpected remote address", Addr: raddr.String()}}
		}
	}

	f := sockaddrToUnix
	if sotype == syscall.SOCK_DGRAM {
		f = sockaddrToUnixgram
	} else if sotype == syscall.SOCK_SEQPACKET {
		f = sockaddrToUnixpacket
	}

	fd, err = socket(net, syscall.AF_UNIX, sotype, 0, la, ra, f)
	if err != nil {
		goto Error
	}
	return fd, nil

Error:
	addr := raddr
	if mode == "listen" {
		addr = laddr
	}
	return nil, &OpError{Op: mode, Net: net, Addr: addr, Err: err}
}

func sockaddrToUnix(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{s.Name, "unix"}
	}
	return nil
}

func sockaddrToUnixgram(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{s.Name, "unixgram"}
	}
	return nil
}

func sockaddrToUnixpacket(sa syscall.Sockaddr) Addr {
	if s, ok := sa.(*syscall.SockaddrUnix); ok {
		return &UnixAddr{s.Name, "unixpacket"}
	}
	return nil
}

func sotypeToNet(sotype int) string {
	switch sotype {
	case syscall.SOCK_STREAM:
		return "unix"
	case syscall.SOCK_SEQPACKET:
		return "unixpacket"
	case syscall.SOCK_DGRAM:
		return "unixgram"
	default:
		panic("sotypeToNet unknown socket type")
	}
	return ""
}

// UnixConn is an implementation of the Conn interface
// for connections to Unix domain sockets.
type UnixConn struct {
	fd *netFD
}

func newUnixConn(fd *netFD) *UnixConn { return &UnixConn{fd} }

func (c *UnixConn) ok() bool { return c != nil && c.fd != nil }

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the Conn Read method.
func (c *UnixConn) Read(b []byte) (n int, err error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	return c.fd.Read(b)
}

// Write implements the Conn Write method.
func (c *UnixConn) Write(b []byte) (n int, err error) {
	if !c.ok() {
		return 0, syscall.EINVAL
	}
	return c.fd.Write(b)
}

// Close closes the Unix domain connection.
func (c *UnixConn) Close() error {
	if !c.ok() {
		return syscall.EINVAL
	}
	err := c.fd.Close()
	c.fd = nil
	return err
}

// LocalAddr returns the local network address, a *UnixAddr.
// Unlike in other protocols, LocalAddr is usually nil for dialed connections.
func (c *UnixConn) LocalAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.laddr
}

// RemoteAddr returns the remote network address, a *UnixAddr.
// Unlike in other protocols, RemoteAddr is usually nil for connections
// accepted by a listener.
func (c *UnixConn) RemoteAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.raddr
}

// SetDeadline implements the Conn SetDeadline method.
func (c *UnixConn) SetDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setDeadline(c.fd, t)
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *UnixConn) SetReadDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setReadDeadline(c.fd, t)
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *UnixConn) SetWriteDeadline(t time.Time) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setWriteDeadline(c.fd, t)
}

// SetReadBuffer sets the size of the operating system's
// receive buffer associated with the connection.
func (c *UnixConn) SetReadBuffer(bytes int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setReadBuffer(c.fd, bytes)
}

// SetWriteBuffer sets the size of the operating system's
// transmit buffer associated with the connection.
func (c *UnixConn) SetWriteBuffer(bytes int) error {
	if !c.ok() {
		return syscall.EINVAL
	}
	return setWriteBuffer(c.fd, bytes)
}

// ReadFromUnix reads a packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromUnix can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetDeadline and SetReadDeadline.
func (c *UnixConn) ReadFromUnix(b []byte) (n int, addr *UnixAddr, err error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, sa, err := c.fd.ReadFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrUnix:
		addr = &UnixAddr{sa.Name, sotypeToNet(c.fd.sotype)}
	}
	return
}

// ReadFrom implements the PacketConn ReadFrom method.
func (c *UnixConn) ReadFrom(b []byte) (n int, addr Addr, err error) {
	if !c.ok() {
		return 0, nil, syscall.EINVAL
	}
	n, uaddr, err := c.ReadFromUnix(b)
	return n, uaddr.toAddr(), err
}

// WriteToUnix writes a packet to addr via c, copying the payload from b.
//
// WriteToUnix can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetDeadline and SetWriteDeadline.
// On packet-oriented connections, write timeouts are rare.
func (c *UnixConn) WriteToUnix(b []byte, addr *UnixAddr) (n int, err error) {
	if !c.ok() {
		return 0, syscall.EINVAL
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

func (c *UnixConn) ReadMsgUnix(b, oob []byte) (n, oobn, flags int, addr *UnixAddr, err error) {
	if !c.ok() {
		return 0, 0, 0, nil, syscall.EINVAL
	}
	n, oobn, flags, sa, err := c.fd.ReadMsg(b, oob)
	switch sa := sa.(type) {
	case *syscall.SockaddrUnix:
		addr = &UnixAddr{sa.Name, sotypeToNet(c.fd.sotype)}
	}
	return
}

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

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (c *UnixConn) File() (f *os.File, err error) { return c.fd.dup() }

// DialUnix connects to the remote address raddr on the network net,
// which must be "unix" or "unixgram".  If laddr is not nil, it is used
// as the local address for the connection.
func DialUnix(net string, laddr, raddr *UnixAddr) (*UnixConn, error) {
	fd, err := unixSocket(net, laddr, raddr, "dial")
	if err != nil {
		return nil, err
	}
	return newUnixConn(fd), nil
}

// UnixListener is a Unix domain socket listener.
// Clients should typically use variables of type Listener
// instead of assuming Unix domain sockets.
type UnixListener struct {
	fd   *netFD
	path string
}

// ListenUnix announces on the Unix domain socket laddr and returns a Unix listener.
// Net must be "unix" (stream sockets).
func ListenUnix(net string, laddr *UnixAddr) (*UnixListener, error) {
	if net != "unix" && net != "unixgram" && net != "unixpacket" {
		return nil, UnknownNetworkError(net)
	}
	if laddr != nil {
		laddr = &UnixAddr{laddr.Name, net} // make our own copy
	}
	fd, err := unixSocket(net, laddr, nil, "listen")
	if err != nil {
		return nil, err
	}
	err = syscall.Listen(fd.sysfd, listenerBacklog)
	if err != nil {
		closesocket(fd.sysfd)
		return nil, &OpError{Op: "listen", Net: net, Addr: laddr, Err: err}
	}
	return &UnixListener{fd, laddr.Name}, nil
}

// AcceptUnix accepts the next incoming call and returns the new connection
// and the remote address.
func (l *UnixListener) AcceptUnix() (*UnixConn, error) {
	if l == nil || l.fd == nil {
		return nil, syscall.EINVAL
	}
	fd, err := l.fd.accept(sockaddrToUnix)
	if err != nil {
		return nil, err
	}
	c := newUnixConn(fd)
	return c, nil
}

// Accept implements the Accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *UnixListener) Accept() (c Conn, err error) {
	c1, err := l.AcceptUnix()
	if err != nil {
		return nil, err
	}
	return c1, nil
}

// Close stops listening on the Unix address.
// Already accepted connections are not closed.
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
	err := l.fd.Close()
	l.fd = nil
	return err
}

// Addr returns the listener's network address.
func (l *UnixListener) Addr() Addr { return l.fd.laddr }

// SetDeadline sets the deadline associated with the listener.
// A zero time value disables the deadline.
func (l *UnixListener) SetDeadline(t time.Time) (err error) {
	if l == nil || l.fd == nil {
		return syscall.EINVAL
	}
	return setDeadline(l.fd, t)
}

// File returns a copy of the underlying os.File, set to blocking mode.
// It is the caller's responsibility to close f when finished.
// Closing c does not affect f, and closing f does not affect c.
func (l *UnixListener) File() (f *os.File, err error) { return l.fd.dup() }

// ListenUnixgram listens for incoming Unix datagram packets addressed to the
// local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send UDP
// packets with per-packet addressing.  The network net must be "unixgram".
func ListenUnixgram(net string, laddr *UnixAddr) (*UDPConn, error) {
	switch net {
	case "unixgram":
	default:
		return nil, UnknownNetworkError(net)
	}
	if laddr == nil {
		return nil, &OpError{"listen", net, nil, errMissingAddress}
	}
	fd, err := unixSocket(net, laddr, nil, "listen")
	if err != nil {
		return nil, err
	}
	return newUDPConn(fd), nil
}

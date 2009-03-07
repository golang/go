// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os";
	"net";
	"strconv";
	"syscall";
)

var (
	BadAddress = os.NewError("malformed address");
	MissingAddress = os.NewError("missing address");
	UnknownNetwork = os.NewError("unknown network");
	UnknownHost = os.NewError("unknown host");
	UnknownSocketFamily = os.NewError("unknown socket family");
)

func LookupHost(name string) (cname string, addrs []string, err *os.Error)

// Split "host:port" into "host" and "port".
// Host cannot contain colons unless it is bracketed.
func splitHostPort(hostport string) (host, port string, err *os.Error) {
	// The port starts after the last colon.
	var i int;
	for i = len(hostport)-1; i >= 0; i-- {
		if hostport[i] == ':' {
			break
		}
	}
	if i < 0 {
		return "", "", BadAddress
	}

	host = hostport[0:i];
	port = hostport[i+1:len(hostport)];

	// Can put brackets around host ...
	if len(host) > 0 && host[0] == '[' && host[len(host)-1] == ']' {
		host = host[1:len(host)-1]
	} else {
		// ... but if there are no brackets, no colons.
		if byteIndex(host, ':') >= 0 {
			return "", "", BadAddress
		}
	}
	return host, port, nil
}

// Join "host" and "port" into "host:port".
// If host contains colons, will join into "[host]:port".
func joinHostPort(host, port string) string {
	// If host has colons, have to bracket it.
	if byteIndex(host, ':') >= 0 {
		return "[" + host + "]:" + port
	}
	return host + ":" + port
}

// Convert "host:port" into IP address and port.
// For now, host and port must be numeric literals.
// Eventually, we'll have name resolution.
func hostPortToIP(net, hostport, mode string) (ip []byte, iport int, err *os.Error) {
	var host, port string;
	host, port, err = splitHostPort(hostport);
	if err != nil {
		return nil, 0, err
	}

	var addr []byte;
	if host == "" {
		if mode == "listen" {
			addr = IPzero;	// wildcard - listen to all
		} else {
			return nil, 0, MissingAddress;
		}
	}

	// Try as an IP address.
	if addr == nil {
		addr = ParseIP(host);
	}
	if addr == nil {
		// Not an IP address.  Try as a DNS name.
		hostname, addrs, err := LookupHost(host);
		if err != nil {
			return nil, 0, err
		}
		if len(addrs) == 0 {
			return nil, 0, UnknownHost
		}
		addr = ParseIP(addrs[0]);
		if addr == nil {
			// should not happen
			return nil, 0, BadAddress
		}
	}

	p, i, ok := dtoi(port, 0);
	if !ok || i != len(port) {
		p, err = LookupPort(net, port);
		if err != nil {
			return nil, 0, err
		}
	}
	if p < 0 || p > 0xFFFF {
		return nil, 0, BadAddress
	}

	return addr, p, nil
}

// Convert socket address into "host:port".
func sockaddrToHostPort(sa *syscall.Sockaddr) (hostport string, err *os.Error) {
	switch sa.Family {
	case syscall.AF_INET, syscall.AF_INET6:
		addr, port, e := sockaddrToIP(sa);
		if e != nil {
			return "", e
		}
		host := addr.String();
		return joinHostPort(host, strconv.Itoa(port)), nil;
	default:
		return "", UnknownSocketFamily
	}
	return "", nil // not reached
}

// Boolean to int.
func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Generic socket creation.
func socket(net, laddr, raddr string, f, p, t int64, la, ra *syscall.Sockaddr)
	(fd *netFD, err *os.Error)
{
	// See ../syscall/exec.go for description of ForkLock.
	syscall.ForkLock.RLock();
	s, e := syscall.Socket(f, p, t);
	if e != 0 {
		syscall.ForkLock.RUnlock();
		return nil, os.ErrnoToError(e)
	}
	syscall.CloseOnExec(s);
	syscall.ForkLock.RUnlock();

	// Allow reuse of recently-used addresses.
	syscall.Setsockopt_int(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1);

	var r int64;
	if la != nil {
		r, e = syscall.Bind(s, la);
		if e != 0 {
			syscall.Close(s);
			return nil, os.ErrnoToError(e)
		}
	}

	if ra != nil {
		r, e = syscall.Connect(s, ra);
		if e != 0 {
			syscall.Close(s);
			return nil, os.ErrnoToError(e)
		}
	}

	fd, err = newFD(s, net, laddr, raddr);
	if err != nil {
		syscall.Close(s);
		return nil, err
	}

	return fd, nil
}


// Generic implementation of Conn interface; not exported.
type connBase struct {
	fd *netFD;
	raddr string;
}

func (c *connBase) FD() *os.FD {
	if c == nil {
		return nil
	}
	return c.fd.osfd;
}

func (c *connBase) sysFD() int64 {
	if c == nil || c.fd == nil {
		return -1;
	}
	return c.fd.fd;
}

func (c *connBase) Read(b []byte) (n int, err *os.Error) {
	n, err = c.fd.Read(b);
	return n, err
}

func (c *connBase) Write(b []byte) (n int, err *os.Error) {
	n, err = c.fd.Write(b);
	return n, err
}

func (c *connBase) ReadFrom(b []byte) (n int, raddr string, err *os.Error) {
	if c == nil {
		return -1, "", os.EINVAL
	}
	n, err = c.Read(b);
	return n, c.raddr, err
}

func (c *connBase) WriteTo(raddr string, b []byte) (n int, err *os.Error) {
	if c == nil {
		return -1, os.EINVAL
	}
	if raddr != c.raddr {
		return -1, os.EINVAL
	}
	n, err = c.Write(b);
	return n, err
}

func (c *connBase) Close() *os.Error {
	if c == nil {
		return os.EINVAL
	}
	return c.fd.Close()
}


func setsockopt_int(fd, level, opt int64, value int) *os.Error {
	return os.ErrnoToError(syscall.Setsockopt_int(fd, level, opt, value));
}

func setsockopt_tv(fd, level, opt int64, nsec int64) *os.Error {
	return os.ErrnoToError(syscall.Setsockopt_tv(fd, level, opt, nsec));
}

func (c *connBase) SetReadBuffer(bytes int) *os.Error {
	return setsockopt_int(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_RCVBUF, bytes);
}

func (c *connBase) SetWriteBuffer(bytes int) *os.Error {
	return setsockopt_int(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_SNDBUF, bytes);
}

func (c *connBase) SetReadTimeout(nsec int64) *os.Error {
	c.fd.rdeadline_delta = nsec;
	return nil;
}

func (c *connBase) SetWriteTimeout(nsec int64) *os.Error {
	c.fd.wdeadline_delta = nsec;
	return nil;
}

func (c *connBase) SetTimeout(nsec int64) *os.Error {
	if e := c.SetReadTimeout(nsec); e != nil {
		return e
	}
	return c.SetWriteTimeout(nsec)
}

func (c *connBase) SetReuseAddr(reuse bool) *os.Error {
	return setsockopt_int(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_REUSEADDR, boolint(reuse));
}

func (c *connBase) BindToDevice(dev string) *os.Error {
	// TODO(rsc): call setsockopt with null-terminated string pointer
	return os.EINVAL
}

func (c *connBase) SetDontRoute(dontroute bool) *os.Error {
	return setsockopt_int(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_DONTROUTE, boolint(dontroute));
}

func (c *connBase) SetKeepAlive(keepalive bool) *os.Error {
	return setsockopt_int(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, boolint(keepalive));
}

func (c *connBase) SetLinger(sec int) *os.Error {
	e := syscall.Setsockopt_linger(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_LINGER, sec);
	return os.ErrnoToError(e);
}


// Internet sockets (TCP, UDP)

// Should we try to use the IPv4 socket interface if we're
// only dealing with IPv4 sockets?  As long as the host system
// understands IPv6, it's okay to pass IPv4 addresses to the IPv6
// interface.  That simplifies our code and is most general.
// If we need to build on a system without IPv6 support, setting
// preferIPv4 here should fall back to the IPv4 socket interface when possible.
const preferIPv4 = false

func internetSocket(net, laddr, raddr string, proto int64, mode string)
	(fd *netFD, err *os.Error)
{
	// Parse addresses (unless they are empty).
	var lip, rip IP;
	var lport, rport int;
	var lerr, rerr *os.Error;

	if laddr != "" {
		lip, lport, lerr = hostPortToIP(net, laddr, mode);
		if lerr != nil {
			return nil, lerr
		}
	}
	if raddr != "" {
		rip, rport, rerr = hostPortToIP(net, raddr, mode);
		if rerr != nil {
			return nil, rerr
		}
	}

	// Figure out IP version.
	// If network has a suffix like "tcp4", obey it.
	vers := 0;
	switch net[len(net)-1] {
	case '4':
		vers = 4;
	case '6':
		vers = 6;
	default:
		// Otherwise, guess.
		// If the addresses are IPv4 and we prefer IPv4, use 4; else 6.
		if preferIPv4 && lip.To4() != nil && rip.To4() != nil {
			vers = 4
		} else {
			vers = 6
		}
	}

	var cvt func(addr []byte, port int) (sa *syscall.Sockaddr, err *os.Error);
	var family int64;
	if vers == 4 {
		cvt = v4ToSockaddr;
		family = syscall.AF_INET
	} else {
		cvt = v6ToSockaddr;
		family = syscall.AF_INET6
	}

	var la, ra *syscall.Sockaddr;
	if lip != nil {
		la, lerr = cvt(lip, lport);
		if lerr != nil {
			return nil, lerr
		}
	}
	if rip != nil {
		ra, rerr = cvt(rip, rport);
		if rerr != nil {
			return nil, rerr
		}
	}

	fd, err = socket(net, laddr, raddr, family, proto, 0, la, ra);
	return fd, err
}


// TCP connections.

type ConnTCP struct {
	connBase
}

func (c *ConnTCP) SetNoDelay(nodelay bool) *os.Error {
	if c == nil {
		return os.EINVAL
	}
	return setsockopt_int(c.sysFD(), syscall.IPPROTO_TCP, syscall.TCP_NODELAY, boolint(nodelay))
}

func newConnTCP(fd *netFD, raddr string) *ConnTCP {
	c := new(ConnTCP);
	c.fd = fd;
	c.raddr = raddr;
	c.SetNoDelay(true);
	return c
}

func DialTCP(net, laddr, raddr string) (c *ConnTCP, err *os.Error) {
	if raddr == "" {
		return nil, MissingAddress
	}
	fd, e := internetSocket(net, laddr, raddr, syscall.SOCK_STREAM, "dial");
	if e != nil {
		return nil, e
	}
	return newConnTCP(fd, raddr), nil
}


// UDP connections.

// TODO(rsc): UDP headers mode

type ConnUDP struct {
	connBase
}

func newConnUDP(fd *netFD, raddr string) *ConnUDP {
	c := new(ConnUDP);
	c.fd = fd;
	c.raddr = raddr;
	return c
}

func DialUDP(net, laddr, raddr string) (c *ConnUDP, err *os.Error) {
	if raddr == "" {
		return nil, MissingAddress
	}
	fd, e := internetSocket(net, laddr, raddr, syscall.SOCK_DGRAM, "dial");
	if e != nil {
		return nil, e
	}
	return newConnUDP(fd, raddr), nil
}


// TODO: raw IP connections

// TODO: raw ethernet connections

// A Conn is a generic network connection.
type Conn interface {
	// Read blocks until data is ready from the connection
	// and then reads into b.  It returns the number
	// of bytes read, or 0 if the connection has been closed.
	Read(b []byte) (n int, err *os.Error);

	// Write writes the data in b to the connection.
	Write(b []byte) (n int, err *os.Error);

	// Close closes the connection.
	Close() *os.Error;

	// For packet-based protocols such as UDP,
	// ReadFrom reads the next packet from the network,
	// returning the number of bytes read and the remote
	// address that sent them.
	ReadFrom(b []byte) (n int, addr string, err *os.Error);

	// For packet-based protocols such as UDP,
	// WriteTo writes the byte buffer b to the network
	// as a single payload, sending it to the target address.
	WriteTo(addr string, b []byte) (n int, err *os.Error);

	// SetReadBuffer sets the size of the operating system's
	// receive buffer associated with the connection.
	SetReadBuffer(bytes int) *os.Error;

	// SetReadBuffer sets the size of the operating system's
	// transmit buffer associated with the connection.
	SetWriteBuffer(bytes int) *os.Error;

	// SetTimeout sets the read and write deadlines associated
	// with the connection.
	SetTimeout(nsec int64) *os.Error;

	// SetReadTimeout sets the time (in nanoseconds) that
	// Read will wait for data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	SetReadTimeout(nsec int64) *os.Error;

	// SetWriteTimeout sets the time (in nanoseconds) that
	// Write will wait to send its data before returning os.EAGAIN.
	// Setting nsec == 0 (the default) disables the deadline.
	// Even if write times out, it may return n > 0, indicating that
	// some of the data was successfully written.
	SetWriteTimeout(nsec int64) *os.Error;

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
	SetLinger(sec int) *os.Error;

	// SetReuseAddr sets whether it is okay to reuse addresses
	// from recent connections that were not properly closed.
	SetReuseAddr(reuseaddr bool) *os.Error;

	// SetDontRoute sets whether outgoing messages should
	// bypass the system routing tables.
	SetDontRoute(dontroute bool) *os.Error;

	// SetKeepAlive sets whether the operating system should send
	// keepalive messages on the connection.
	SetKeepAlive(keepalive bool) *os.Error;

	// BindToDevice binds a connection to a particular network device.
	BindToDevice(dev string) *os.Error;
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
func Dial(net, laddr, raddr string) (c Conn, err *os.Error) {
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
/*
	case "ether":
		c, err := DialEther(net, laddr, raddr);
		return c, err;
	case "ipv4":
		c, err := DialIPv4(net, laddr, raddr);
		return c, err;
	case "ipv6":
		c, err := DialIPv6(net, laddr, raddr);
		return c, err
*/
	}
	return nil, UnknownNetwork
}

// A Listener is a generic network listener.
// Accept waits for the next connection and Close closes the connection.
type Listener interface {
	Accept() (c Conn, raddr string, err *os.Error);
	Close() *os.Error;
}

// ListenerTCP is a TCP network listener.
// Clients should typically use variables of type Listener
// instead of assuming TCP.
type ListenerTCP struct {
	fd *netFD;
	laddr string
}

// ListenTCP announces on the TCP address laddr and returns a TCP listener.
// Net must be "tcp", "tcp4", or "tcp6".
func ListenTCP(net, laddr string) (l *ListenerTCP, err *os.Error) {
	fd, e := internetSocket(net, laddr, "", syscall.SOCK_STREAM, "listen");
	if e != nil {
		return nil, e
	}
	r, e1 := syscall.Listen(fd.fd, listenBacklog());
	if e1 != 0 {
		syscall.Close(fd.fd);
		return nil, os.ErrnoToError(e1)
	}
	l = new(ListenerTCP);
	l.fd = fd;
	return l, nil
}

// AcceptTCP accepts the next incoming call and returns the new connection
// and the remote address.
func (l *ListenerTCP) AcceptTCP() (c *ConnTCP, raddr string, err *os.Error) {
	if l == nil || l.fd == nil || l.fd.fd < 0 {
		return nil, "", os.EINVAL
	}
	var sa syscall.Sockaddr;
	fd, e := l.fd.Accept(&sa);
	if e != nil {
		return nil, "", e
	}
	raddr, err = sockaddrToHostPort(&sa);
	if err != nil {
		fd.Close();
		return nil, "", err
	}
	return newConnTCP(fd, raddr), raddr, nil
}

// Accept implements the accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *ListenerTCP) Accept() (c Conn, raddr string, err *os.Error) {
	c1, r1, e1 := l.AcceptTCP();
	if e1 != nil {
		return nil, "", e1
	}
	return c1, r1, nil
}

// Close stops listening on the TCP address.
// Already Accepted connections are not closed.
func (l *ListenerTCP) Close() *os.Error {
	if l == nil || l.fd == nil {
		return os.EINVAL
	}
	return l.fd.Close()
}

// Listen announces on the local network address laddr.
// The network string net must be "tcp", "tcp4", or "tcp6".
func Listen(net, laddr string) (l Listener, err *os.Error) {
	switch net {
	case "tcp", "tcp4", "tcp6":
		l, err := ListenTCP(net, laddr);
		if err != nil {
			return nil, err
		}
		return l, nil
/*
	more here
*/
	// BUG(rsc): Listen should support UDP.
	}
	return nil, UnknownNetwork
}


// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"net";
	"os";
	"strconv";
	"syscall";
)

var (
	BadAddress os.Error = &Error{"malformed address"};
	MissingAddress os.Error = &Error{"missing address"};
	UnknownNetwork os.Error = &Error{"unknown network"};
	UnknownHost os.Error = &Error{"unknown host"};
	UnknownSocketFamily os.Error = &Error{"unknown socket family"};
)


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

// Should we try to use the IPv4 socket interface if we're
// only dealing with IPv4 sockets?  As long as the host system
// understands IPv6, it's okay to pass IPv4 addresses to the IPv6
// interface.  That simplifies our code and is most general.
// Unfortunately, we need to run on kernels built without IPv6 support too.
// So probe the kernel to figure it out.
func kernelSupportsIPv6() bool {
	fd, e := syscall.Socket(syscall.AF_INET6, syscall.SOCK_STREAM, syscall.IPPROTO_TCP);
	if fd >= 0 {
		syscall.Close(fd)
	}
	return e == 0
}

var preferIPv4 = !kernelSupportsIPv6()

// TODO(rsc): if syscall.OS == "linux", we're supposd to read
// /proc/sys/net/core/somaxconn,
// to take advantage of kernels that have raised the limit.
func listenBacklog() int {
	return syscall.SOMAXCONN
}

func LookupHost(name string) (cname string, addrs []string, err os.Error)

// Split "host:port" into "host" and "port".
// Host cannot contain colons unless it is bracketed.
func splitHostPort(hostport string) (host, port string, err os.Error) {
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
func hostPortToIP(net, hostport, mode string) (ip IP, iport int, err os.Error) {
	var host, port string;
	host, port, err = splitHostPort(hostport);
	if err != nil {
		return nil, 0, err
	}

	var addr IP;
	if host == "" {
		if mode == "listen" {
			if preferIPv4 {
				addr = IPv4zero;
			} else {
				addr = IPzero;	// wildcard - listen to all
			}
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

func sockaddrToString(sa syscall.Sockaddr) (name string, err os.Error) {
	switch a := sa.(type) {
	case *syscall.SockaddrInet4:
		return joinHostPort(IP(&a.Addr).String(), strconv.Itoa(a.Port)), nil;
	case *syscall.SockaddrInet6:
		return joinHostPort(IP(&a.Addr).String(), strconv.Itoa(a.Port)), nil;
	case *syscall.SockaddrUnix:
		return a.Name, nil;
	}
	return "", UnknownSocketFamily
}

func ipToSockaddr(family int, ip IP, port int) (syscall.Sockaddr, os.Error) {
	switch family {
	case syscall.AF_INET:
		if ip = ip.To4(); ip == nil {
			return nil, os.EINVAL
		}
		s := new(syscall.SockaddrInet4);
		for i := 0; i < IPv4len; i++ {
			s.Addr[i] = ip[i];
		}
		s.Port = port;
		return s, nil;
	case syscall.AF_INET6:
		// IPv4 callers use 0.0.0.0 to mean "announce on any available address".
		// In IPv6 mode, Linux treats that as meaning "announce on 0.0.0.0",
		// which it refuses to do.  Rewrite to the IPv6 all zeros.
		if p4 := ip.To4(); p4 != nil && p4[0] == 0 && p4[1] == 0 && p4[2] == 0 && p4[3] == 0 {
			ip = IPzero;
		}
		if ip = ip.To16(); ip == nil {
			return nil, os.EINVAL
		}
		s := new(syscall.SockaddrInet6);
		for i := 0; i < IPv6len; i++ {
			s.Addr[i] = ip[i];
		}
		s.Port = port;
		return s, nil;
	}
	return nil, os.EINVAL;
}

// Boolean to int.
func boolint(b bool) int {
	if b {
		return 1
	}
	return 0
}

// Generic socket creation.
func socket(net, laddr, raddr string, f, p, t int, la, ra syscall.Sockaddr) (fd *netFD, err os.Error) {
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
	syscall.SetsockoptInt(s, syscall.SOL_SOCKET, syscall.SO_REUSEADDR, 1);

	var r int64;
	if la != nil {
		e = syscall.Bind(s, la);
		if e != 0 {
			syscall.Close(s);
			return nil, os.ErrnoToError(e)
		}
	}

	if ra != nil {
		e = syscall.Connect(s, ra);
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

func (c *connBase) LocalAddr() string {
	if c == nil {
		return ""
	}
	return c.fd.addr();
}

func (c *connBase) RemoteAddr() string {
	if c == nil {
		return ""
	}
	return c.fd.remoteAddr();
}

func (c *connBase) File() *os.File {
	if c == nil {
		return nil
	}
	return c.fd.file;
}

func (c *connBase) sysFD() int {
	if c == nil || c.fd == nil {
		return -1;
	}
	return c.fd.fd;
}

func (c *connBase) Read(b []byte) (n int, err os.Error) {
	n, err = c.fd.Read(b);
	return n, err
}

func (c *connBase) Write(b []byte) (n int, err os.Error) {
	n, err = c.fd.Write(b);
	return n, err
}

func (c *connBase) ReadFrom(b []byte) (n int, raddr string, err os.Error) {
	if c == nil {
		return -1, "", os.EINVAL
	}
	n, err = c.Read(b);
	return n, c.raddr, err
}

func (c *connBase) WriteTo(raddr string, b []byte) (n int, err os.Error) {
	if c == nil {
		return -1, os.EINVAL
	}
	if raddr != c.raddr {
		return -1, os.EINVAL
	}
	n, err = c.Write(b);
	return n, err
}

func (c *connBase) Close() os.Error {
	if c == nil {
		return os.EINVAL
	}
	return c.fd.Close()
}


func setsockoptInt(fd, level, opt int, value int) os.Error {
	return os.ErrnoToError(syscall.SetsockoptInt(fd, level, opt, value));
}

func setsockoptNsec(fd, level, opt int, nsec int64) os.Error {
	var tv = syscall.NsecToTimeval(nsec);
	return os.ErrnoToError(syscall.SetsockoptTimeval(fd, level, opt, &tv));
}

func (c *connBase) SetReadBuffer(bytes int) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_RCVBUF, bytes);
}

func (c *connBase) SetWriteBuffer(bytes int) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_SNDBUF, bytes);
}

func (c *connBase) SetReadTimeout(nsec int64) os.Error {
	c.fd.rdeadline_delta = nsec;
	return nil;
}

func (c *connBase) SetWriteTimeout(nsec int64) os.Error {
	c.fd.wdeadline_delta = nsec;
	return nil;
}

func (c *connBase) SetTimeout(nsec int64) os.Error {
	if e := c.SetReadTimeout(nsec); e != nil {
		return e
	}
	return c.SetWriteTimeout(nsec)
}

func (c *connBase) SetReuseAddr(reuse bool) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_REUSEADDR, boolint(reuse));
}

func (c *connBase) BindToDevice(dev string) os.Error {
	// TODO(rsc): call setsockopt with null-terminated string pointer
	return os.EINVAL
}

func (c *connBase) SetDontRoute(dontroute bool) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_DONTROUTE, boolint(dontroute));
}

func (c *connBase) SetKeepAlive(keepalive bool) os.Error {
	return setsockoptInt(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_KEEPALIVE, boolint(keepalive));
}

func (c *connBase) SetLinger(sec int) os.Error {
	var l syscall.Linger;
	if sec >= 0 {
		l.Onoff = 1;
		l.Linger = int32(sec);
	} else {
		l.Onoff = 0;
		l.Linger = 0;
	}
	e := syscall.SetsockoptLinger(c.sysFD(), syscall.SOL_SOCKET, syscall.SO_LINGER, &l);
	return os.ErrnoToError(e);
}


// Internet sockets (TCP, UDP)

func internetSocket(net, laddr, raddr string, proto int, mode string) (fd *netFD, err os.Error) {
	// Parse addresses (unless they are empty).
	var lip, rip IP;
	var lport, rport int;

	if laddr != "" {
		if lip, lport, err = hostPortToIP(net, laddr, mode); err != nil {
			return
		}
	}
	if raddr != "" {
		if rip, rport, err = hostPortToIP(net, raddr, mode); err != nil {
			return
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
		if preferIPv4 && (lip == nil || lip.To4() != nil) && (rip == nil || rip.To4() != nil) {
			vers = 4
		} else {
			vers = 6
		}
	}

	var family int;
	if vers == 4 {
		family = syscall.AF_INET
	} else {
		family = syscall.AF_INET6
	}

	var la, ra syscall.Sockaddr;
	if lip != nil {
		if la, err = ipToSockaddr(family, lip, lport); err != nil {
			return
		}
	}
	if rip != nil {
		if ra, err = ipToSockaddr(family, rip, rport); err != nil {
			return
		}
	}

	fd, err = socket(net, laddr, raddr, family, proto, 0, la, ra);
	return fd, err
}


// TCP connections.

// ConnTCP is an implementation of the Conn interface
// for TCP network connections.
type ConnTCP struct {
	connBase
}

func (c *ConnTCP) SetNoDelay(nodelay bool) os.Error {
	if c == nil {
		return os.EINVAL
	}
	return setsockoptInt(c.sysFD(), syscall.IPPROTO_TCP, syscall.TCP_NODELAY, boolint(nodelay))
}

func newConnTCP(fd *netFD, raddr string) *ConnTCP {
	c := new(ConnTCP);
	c.fd = fd;
	c.raddr = raddr;
	c.SetNoDelay(true);
	return c
}

// DialTCP is like Dial but can only connect to TCP networks
// and returns a ConnTCP structure.
func DialTCP(net, laddr, raddr string) (c *ConnTCP, err os.Error) {
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

// ConnUDP is an implementation of the Conn interface
// for UDP network connections.
type ConnUDP struct {
	connBase
}

func newConnUDP(fd *netFD, raddr string) *ConnUDP {
	c := new(ConnUDP);
	c.fd = fd;
	c.raddr = raddr;
	return c
}

// DialUDP is like Dial but can only connect to UDP networks
// and returns a ConnUDP structure.
func DialUDP(net, laddr, raddr string) (c *ConnUDP, err os.Error) {
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


// Unix domain sockets

func unixSocket(net, laddr, raddr string, mode string) (fd *netFD, err os.Error) {
	var proto int;
	switch net {
	default:
		return nil, UnknownNetwork;
	case "unix":
		proto = syscall.SOCK_STREAM;
	case "unix-dgram":
		proto = syscall.SOCK_DGRAM;
	}

	var la, ra syscall.Sockaddr;
	switch mode {
	case "dial":
		if laddr != "" {
			return nil, BadAddress;
		}
		if raddr == "" {
			return nil, MissingAddress;
		}
		ra = &syscall.SockaddrUnix{Name: raddr};

	case "listen":
		if laddr == "" {
			return nil, MissingAddress;
		}
		la = &syscall.SockaddrUnix{Name: laddr};
		if raddr != "" {
			return nil, BadAddress;
		}
	}

	fd, err = socket(net, laddr, raddr, syscall.AF_UNIX, proto, 0, la, ra);
	return fd, err
}

// ConnUnix is an implementation of the Conn interface
// for connections to Unix domain sockets.
type ConnUnix struct {
	connBase
}

func newConnUnix(fd *netFD, raddr string) *ConnUnix {
	c := new(ConnUnix);
	c.fd = fd;
	c.raddr = raddr;
	return c;
}

// DialUnix is like Dial but can only connect to Unix domain sockets
// and returns a ConnUnix structure.  The laddr argument must be
// the empty string; it is included only to match the signature of
// the other dial routines.
func DialUnix(net, laddr, raddr string) (c *ConnUnix, err os.Error) {
	fd, e := unixSocket(net, laddr, raddr, "dial");
	if e != nil {
		return nil, e
	}
	return newConnUnix(fd, raddr), nil;
}

// ListenerUnix is a Unix domain socket listener.
// Clients should typically use variables of type Listener
// instead of assuming Unix domain sockets.
type ListenerUnix struct {
	fd *netFD;
	laddr string
}

// ListenUnix announces on the Unix domain socket laddr and returns a Unix listener.
// Net can be either "unix" (stream sockets) or "unix-dgram" (datagram sockets).
func ListenUnix(net, laddr string) (l *ListenerUnix, err os.Error) {
	fd, e := unixSocket(net, laddr, "", "listen");
	if e != nil {
		// Check for socket ``in use'' but ``refusing connections,''
		// which means some program created it and exited
		// without unlinking it from the file system.
		// Clean up on that program's behalf and try again.
		// Don't do this for Linux's ``abstract'' sockets, which begin with @.
		if e != os.EADDRINUSE || laddr[0] == '@' {
			return nil, e;
		}
		fd1, e1 := unixSocket(net, "", laddr, "dial");
		if e1 == nil {
			fd1.Close();
		}
		if e1 != os.ECONNREFUSED {
			return nil, e;
		}
		syscall.Unlink(laddr);
		fd1, e1 = unixSocket(net, laddr, "", "listen");
		if e1 != nil {
			return nil, e;
		}
		fd = fd1;
	}
	e1 := syscall.Listen(fd.fd, 8); // listenBacklog());
	if e1 != 0 {
		syscall.Close(fd.fd);
		return nil, os.ErrnoToError(e1);
	}
	return &ListenerUnix{fd, laddr}, nil;
}

// AcceptUnix accepts the next incoming call and returns the new connection
// and the remote address.
func (l *ListenerUnix) AcceptUnix() (c *ConnUnix, raddr string, err os.Error) {
	if l == nil || l.fd == nil || l.fd.fd < 0 {
		return nil, "", os.EINVAL
	}
	fd, e := l.fd.accept();
	if e != nil {
		return nil, "", e
	}
	return newConnUnix(fd, fd.raddr), raddr, nil
}

// Accept implements the Accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *ListenerUnix) Accept() (c Conn, raddr string, err os.Error) {
	// TODO(rsc): 6g bug prevents saying
	//	c, raddr, err = l.AcceptUnix();
	//	return;
	c1, r1, e1 := l.AcceptUnix();
	return c1, r1, e1;
}


// Close stops listening on the Unix address.
// Already accepted connections are not closed.
func (l *ListenerUnix) Close() os.Error {
	if l == nil || l.fd == nil {
		return os.EINVAL
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
	if l.laddr[0] != '@' {
		syscall.Unlink(l.laddr);
	}
	err := l.fd.Close();
	l.fd = nil;
	return err;
}

// Addr returns the listener's network address.
func (l *ListenerUnix) Addr() string {
	return l.fd.addr();
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
	Accept() (c Conn, raddr string, err os.Error);
	Close() os.Error;
	Addr() string;	// Listener's network address
}

// ListenerTCP is a TCP network listener.
// Clients should typically use variables of type Listener
// instead of assuming TCP.
type ListenerTCP struct {
	fd *netFD;
}

// ListenTCP announces on the TCP address laddr and returns a TCP listener.
// Net must be "tcp", "tcp4", or "tcp6".
// If laddr has a port of 0, it means to listen on some available port.
// The caller can use l.Addr() to retrieve the chosen address.
func ListenTCP(net, laddr string) (l *ListenerTCP, err os.Error) {
	fd, e := internetSocket(net, laddr, "", syscall.SOCK_STREAM, "listen");
	if e != nil {
		return nil, e
	}
	e1 := syscall.Listen(fd.fd, listenBacklog());
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
func (l *ListenerTCP) AcceptTCP() (c *ConnTCP, raddr string, err os.Error) {
	if l == nil || l.fd == nil || l.fd.fd < 0 {
		return nil, "", os.EINVAL
	}
	fd, e := l.fd.accept();
	if e != nil {
		return nil, "", e
	}
	return newConnTCP(fd, fd.raddr), fd.raddr, nil
}

// Accept implements the Accept method in the Listener interface;
// it waits for the next call and returns a generic Conn.
func (l *ListenerTCP) Accept() (c Conn, raddr string, err os.Error) {
	c1, r1, e1 := l.AcceptTCP();
	if e1 != nil {
		return nil, "", e1
	}
	return c1, r1, nil
}

// Close stops listening on the TCP address.
// Already Accepted connections are not closed.
func (l *ListenerTCP) Close() os.Error {
	if l == nil || l.fd == nil {
		return os.EINVAL
	}
	return l.fd.Close()
}

// Addr returns the listener's network address.
func (l *ListenerTCP) Addr() string {
	return l.fd.addr();
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
/*
	more here
*/
	// BUG(rsc): Listen should support UDP.
	}
	return nil, UnknownNetwork
}


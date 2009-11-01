// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP sockets

package net

import (
	"os";
	"syscall";
)

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
		return nil, &OpError{"listen", "tcp", laddr, os.Errno(e1)};
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

// Internet sockets (TCP, UDP)

func internetSocket(net, laddr, raddr string, proto int, mode string) (fd *netFD, err os.Error) {
	// Parse addresses (unless they are empty).
	var lip, rip IP;
	var lport, rport int;

	if laddr != "" {
		if lip, lport, err = hostPortToIP(net, laddr, mode); err != nil {
			goto Error;
		}
	}
	if raddr != "" {
		if rip, rport, err = hostPortToIP(net, raddr, mode); err != nil {
			goto Error;
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
			goto Error;
		}
	}
	if rip != nil {
		if ra, err = ipToSockaddr(family, rip, rport); err != nil {
			goto Error;
		}
	}

	fd, err = socket(net, laddr, raddr, family, proto, 0, la, ra);
	if err != nil {
		goto Error;
	}
	return fd, nil;

Error:
	addr := raddr;
	if mode == "listen" {
		addr = laddr;
	}
	return nil, &OpError{mode, net, addr, err};
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
		return nil, &OpError{"dial", "tcp", "", errMissingAddress}
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
		return nil, &OpError{"dial", "udp", "", errMissingAddress}
	}
	fd, e := internetSocket(net, laddr, raddr, syscall.SOCK_DGRAM, "dial");
	if e != nil {
		return nil, e
	}
	return newConnUDP(fd, raddr), nil
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

// Split "host:port" into "host" and "port".
// Host cannot contain colons unless it is bracketed.
func splitHostPort(hostport string) (host, port string, err os.Error) {
	// The port starts after the last colon.
	i := last(hostport, ':');
	if i < 0 {
		err = &AddrError{"missing port in address", hostport};
		return;
	}

	host, port = hostport[0:i], hostport[i+1:len(hostport)];

	// Can put brackets around host ...
	if len(host) > 0 && host[0] == '[' && host[len(host)-1] == ']' {
		host = host[1:len(host)-1]
	} else {
		// ... but if there are no brackets, no colons.
		if byteIndex(host, ':') >= 0 {
			err = &AddrError{"too many colons in address", hostport};
			return;
		}
	}
	return;
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
	host, port, err := splitHostPort(hostport);
	if err != nil {
		goto Error;
	}

	var addr IP;
	if host == "" {
		if mode != "listen" {
			err = &AddrError{"no host in address", hostport};
			goto Error;
		}
		if preferIPv4 {
			addr = IPv4zero;
		} else {
			addr = IPzero;	// wildcard - listen to all
		}
	}

	// Try as an IP address.
	if addr == nil {
		addr = ParseIP(host);
	}
	if addr == nil {
		// Not an IP address.  Try as a DNS name.
		_, addrs, err1 := LookupHost(host);
		if err1 != nil {
			err = err1;
			goto Error;
		}
		addr = ParseIP(addrs[0]);
		if addr == nil {
			// should not happen
			err = &AddrError{"LookupHost returned invalid address", addrs[0]};
			goto Error;
		}
	}

	p, i, ok := dtoi(port, 0);
	if !ok || i != len(port) {
		p, err = LookupPort(net, port);
		if err != nil {
			goto Error;
		}
	}
	if p < 0 || p > 0xFFFF {
		err = &AddrError{"invalid port", port};
		goto Error;
	}

	return addr, p, nil;

Error:
	return nil, 0, err;
}


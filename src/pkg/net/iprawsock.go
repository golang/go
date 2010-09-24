// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// (Raw) IP sockets

package net

import (
	"os"
	"sync"
	"syscall"
)

var onceReadProtocols sync.Once

func sockaddrToIP(sa syscall.Sockaddr) Addr {
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		return &IPAddr{sa.Addr[0:]}
	case *syscall.SockaddrInet6:
		return &IPAddr{sa.Addr[0:]}
	}
	return nil
}

// IPAddr represents the address of a IP end point.
type IPAddr struct {
	IP IP
}

// Network returns the address's network name, "ip".
func (a *IPAddr) Network() string { return "ip" }

func (a *IPAddr) String() string {
	if a == nil {
		return "<nil>"
	}
	return a.IP.String()
}

func (a *IPAddr) family() int {
	if a == nil || len(a.IP) <= 4 {
		return syscall.AF_INET
	}
	if ip := a.IP.To4(); ip != nil {
		return syscall.AF_INET
	}
	return syscall.AF_INET6
}

func (a *IPAddr) sockaddr(family int) (syscall.Sockaddr, os.Error) {
	return ipToSockaddr(family, a.IP, 0)
}

func (a *IPAddr) toAddr() sockaddr {
	if a == nil { // nil *IPAddr
		return nil // nil interface
	}
	return a
}

// ResolveIPAddr parses addr as a IP address and resolves domain
// names to numeric addresses.  A literal IPv6 host address must be
// enclosed in square brackets, as in "[::]".
func ResolveIPAddr(addr string) (*IPAddr, os.Error) {
	ip, err := hostToIP(addr)
	if err != nil {
		return nil, err
	}
	return &IPAddr{ip}, nil
}

// IPConn is the implementation of the Conn and PacketConn
// interfaces for IP network connections.
type IPConn struct {
	fd *netFD
}

func newIPConn(fd *netFD) *IPConn { return &IPConn{fd} }

func (c *IPConn) ok() bool { return c != nil && c.fd != nil }

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the net.Conn Read method.
func (c *IPConn) Read(b []byte) (n int, err os.Error) {
	n, _, err = c.ReadFrom(b)
	return
}

// Write implements the net.Conn Write method.
func (c *IPConn) Write(b []byte) (n int, err os.Error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	return c.fd.Write(b)
}

// Close closes the IP connection.
func (c *IPConn) Close() os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	err := c.fd.Close()
	c.fd = nil
	return err
}

// LocalAddr returns the local network address.
func (c *IPConn) LocalAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.laddr
}

// RemoteAddr returns the remote network address, a *IPAddr.
func (c *IPConn) RemoteAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.fd.raddr
}

// SetTimeout implements the net.Conn SetTimeout method.
func (c *IPConn) SetTimeout(nsec int64) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setTimeout(c.fd, nsec)
}

// SetReadTimeout implements the net.Conn SetReadTimeout method.
func (c *IPConn) SetReadTimeout(nsec int64) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadTimeout(c.fd, nsec)
}

// SetWriteTimeout implements the net.Conn SetWriteTimeout method.
func (c *IPConn) SetWriteTimeout(nsec int64) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteTimeout(c.fd, nsec)
}

// SetReadBuffer sets the size of the operating system's
// receive buffer associated with the connection.
func (c *IPConn) SetReadBuffer(bytes int) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setReadBuffer(c.fd, bytes)
}

// SetWriteBuffer sets the size of the operating system's
// transmit buffer associated with the connection.
func (c *IPConn) SetWriteBuffer(bytes int) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	return setWriteBuffer(c.fd, bytes)
}

// IP-specific methods.

// ReadFromIP reads a IP packet from c, copying the payload into b.
// It returns the number of bytes copied into b and the return address
// that was on the packet.
//
// ReadFromIP can be made to time out and return an error with
// Timeout() == true after a fixed time limit; see SetTimeout and
// SetReadTimeout.
func (c *IPConn) ReadFromIP(b []byte) (n int, addr *IPAddr, err os.Error) {
	if !c.ok() {
		return 0, nil, os.EINVAL
	}
	// TODO(cw,rsc): consider using readv if we know the family
	// type to avoid the header trim/copy
	n, sa, err := c.fd.ReadFrom(b)
	switch sa := sa.(type) {
	case *syscall.SockaddrInet4:
		addr = &IPAddr{sa.Addr[0:]}
		if len(b) >= 4 { // discard ipv4 header
			hsize := (int(b[0]) & 0xf) * 4
			copy(b, b[hsize:])
			n -= hsize
		}
	case *syscall.SockaddrInet6:
		addr = &IPAddr{sa.Addr[0:]}
	}
	return
}

// ReadFrom implements the net.PacketConn ReadFrom method.
func (c *IPConn) ReadFrom(b []byte) (n int, addr Addr, err os.Error) {
	if !c.ok() {
		return 0, nil, os.EINVAL
	}
	n, uaddr, err := c.ReadFromIP(b)
	return n, uaddr.toAddr(), err
}

// WriteToIP writes a IP packet to addr via c, copying the payload from b.
//
// WriteToIP can be made to time out and return
// an error with Timeout() == true after a fixed time limit;
// see SetTimeout and SetWriteTimeout.
// On packet-oriented connections, write timeouts are rare.
func (c *IPConn) WriteToIP(b []byte, addr *IPAddr) (n int, err os.Error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	sa, err1 := addr.sockaddr(c.fd.family)
	if err1 != nil {
		return 0, &OpError{Op: "write", Net: "ip", Addr: addr, Error: err1}
	}
	return c.fd.WriteTo(b, sa)
}

// WriteTo implements the net.PacketConn WriteTo method.
func (c *IPConn) WriteTo(b []byte, addr Addr) (n int, err os.Error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	a, ok := addr.(*IPAddr)
	if !ok {
		return 0, &OpError{"writeto", "ip", addr, os.EINVAL}
	}
	return c.WriteToIP(b, a)
}

// Convert "host" into IP address.
func hostToIP(host string) (ip IP, err os.Error) {
	var addr IP
	// Try as an IP address.
	addr = ParseIP(host)
	if addr == nil {
		// Not an IP address.  Try as a DNS name.
		_, addrs, err1 := LookupHost(host)
		if err1 != nil {
			err = err1
			goto Error
		}
		addr = ParseIP(addrs[0])
		if addr == nil {
			// should not happen
			err = &AddrError{"LookupHost returned invalid address", addrs[0]}
			goto Error
		}
	}

	return addr, nil

Error:
	return nil, err
}


var protocols map[string]int

func readProtocols() {
	protocols = make(map[string]int)
	if file, err := open("/etc/protocols"); err == nil {
		for line, ok := file.readLine(); ok; line, ok = file.readLine() {
			// tcp    6   TCP    # transmission control protocol
			if i := byteIndex(line, '#'); i >= 0 {
				line = line[0:i]
			}
			f := getFields(line)
			if len(f) < 2 {
				continue
			}
			if proto, _, ok := dtoi(f[1], 0); ok {
				protocols[f[0]] = proto
				for _, alias := range f[2:] {
					protocols[alias] = proto
				}
			}
		}
		file.close()
	}
}

func netProtoSplit(netProto string) (net string, proto int, err os.Error) {
	onceReadProtocols.Do(readProtocols)
	i := last(netProto, ':')
	if i < 0 { // no colon
		return "", 0, os.ErrorString("no IP protocol specified")
	}
	net = netProto[0:i]
	protostr := netProto[i+1:]
	proto, i, ok := dtoi(protostr, 0)
	if !ok || i != len(protostr) {
		// lookup by name
		proto, ok = protocols[protostr]
		if ok {
			return
		}
	}
	return
}

// DialIP connects to the remote address raddr on the network net,
// which must be "ip", "ip4", or "ip6".
func DialIP(netProto string, laddr, raddr *IPAddr) (c *IPConn, err os.Error) {
	net, proto, err := netProtoSplit(netProto)
	if err != nil {
		return
	}
	switch prefixBefore(net, ':') {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(net)
	}
	if raddr == nil {
		return nil, &OpError{"dial", "ip", nil, errMissingAddress}
	}
	fd, e := internetSocket(net, laddr.toAddr(), raddr.toAddr(), syscall.SOCK_RAW, proto, "dial", sockaddrToIP)
	if e != nil {
		return nil, e
	}
	return newIPConn(fd), nil
}

// ListenIP listens for incoming IP packets addressed to the
// local address laddr.  The returned connection c's ReadFrom
// and WriteTo methods can be used to receive and send IP
// packets with per-packet addressing.
func ListenIP(netProto string, laddr *IPAddr) (c *IPConn, err os.Error) {
	net, proto, err := netProtoSplit(netProto)
	if err != nil {
		return
	}
	switch prefixBefore(net, ':') {
	case "ip", "ip4", "ip6":
	default:
		return nil, UnknownNetworkError(net)
	}
	fd, e := internetSocket(net, laddr.toAddr(), nil, syscall.SOCK_RAW, proto, "dial", sockaddrToIP)
	if e != nil {
		return nil, e
	}
	return newIPConn(fd), nil
}

// BindToDevice binds an IPConn to a network interface.
func (c *IPConn) BindToDevice(device string) os.Error {
	if !c.ok() {
		return os.EINVAL
	}
	c.fd.incref()
	defer c.fd.decref()
	return os.NewSyscallError("setsockopt", syscall.BindToDevice(c.fd.sysfd, device))
}

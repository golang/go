// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP sockets stubs for Plan 9

package net

import (
	"errors"
	"io"
	"os"
	"time"
)

// probeIPv6Stack returns two boolean values.  If the first boolean value is
// true, kernel supports basic IPv6 functionality.  If the second
// boolean value is true, kernel supports IPv6 IPv4-mapping.
func probeIPv6Stack() (supportsIPv6, supportsIPv4map bool) {
	return false, false
}

// parsePlan9Addr parses address of the form [ip!]port (e.g. 127.0.0.1!80).
func parsePlan9Addr(s string) (ip IP, iport int, err error) {
	addr := IPv4zero // address contains port only
	i := byteIndex(s, '!')
	if i >= 0 {
		addr = ParseIP(s[:i])
		if addr == nil {
			return nil, 0, errors.New("net: parsing IP failed")
		}
	}
	p, _, ok := dtoi(s[i+1:], 0)
	if !ok {
		return nil, 0, errors.New("net: parsing port failed")
	}
	if p < 0 || p > 0xFFFF {
		return nil, 0, &AddrError{"invalid port", string(p)}
	}
	return addr, p, nil
}

func readPlan9Addr(proto, filename string) (addr Addr, err error) {
	var buf [128]byte

	f, err := os.Open(filename)
	if err != nil {
		return
	}
	n, err := f.Read(buf[:])
	if err != nil {
		return
	}
	ip, port, err := parsePlan9Addr(string(buf[:n]))
	if err != nil {
		return
	}
	switch proto {
	case "tcp":
		addr = &TCPAddr{ip, port}
	case "udp":
		addr = &UDPAddr{ip, port}
	default:
		return nil, errors.New("unknown protocol " + proto)
	}
	return addr, nil
}

type plan9Conn struct {
	proto, name, dir string
	ctl, data        *os.File
	laddr, raddr     Addr
}

func newPlan9Conn(proto, name string, ctl *os.File, laddr, raddr Addr) *plan9Conn {
	return &plan9Conn{proto, name, "/net/" + proto + "/" + name, ctl, nil, laddr, raddr}
}

func (c *plan9Conn) ok() bool { return c != nil && c.ctl != nil }

// Implementation of the Conn interface - see Conn for documentation.

// Read implements the Conn Read method.
func (c *plan9Conn) Read(b []byte) (n int, err error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	if c.data == nil {
		c.data, err = os.OpenFile(c.dir+"/data", os.O_RDWR, 0)
		if err != nil {
			return 0, err
		}
	}
	n, err = c.data.Read(b)
	if c.proto == "udp" && err == io.EOF {
		n = 0
		err = nil
	}
	return
}

// Write implements the Conn Write method.
func (c *plan9Conn) Write(b []byte) (n int, err error) {
	if !c.ok() {
		return 0, os.EINVAL
	}
	if c.data == nil {
		c.data, err = os.OpenFile(c.dir+"/data", os.O_RDWR, 0)
		if err != nil {
			return 0, err
		}
	}
	return c.data.Write(b)
}

// Close closes the connection.
func (c *plan9Conn) Close() error {
	if !c.ok() {
		return os.EINVAL
	}
	err := c.ctl.Close()
	if err != nil {
		return err
	}
	if c.data != nil {
		err = c.data.Close()
	}
	c.ctl = nil
	c.data = nil
	return err
}

// LocalAddr returns the local network address.
func (c *plan9Conn) LocalAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.laddr
}

// RemoteAddr returns the remote network address.
func (c *plan9Conn) RemoteAddr() Addr {
	if !c.ok() {
		return nil
	}
	return c.raddr
}

// SetDeadline implements the Conn SetDeadline method.
func (c *plan9Conn) SetDeadline(t time.Time) error {
	return os.EPLAN9
}

// SetReadDeadline implements the Conn SetReadDeadline method.
func (c *plan9Conn) SetReadDeadline(t time.Time) error {
	return os.EPLAN9
}

// SetWriteDeadline implements the Conn SetWriteDeadline method.
func (c *plan9Conn) SetWriteDeadline(t time.Time) error {
	return os.EPLAN9
}

func startPlan9(net string, addr Addr) (ctl *os.File, dest, proto, name string, err error) {
	var (
		ip   IP
		port int
	)
	switch a := addr.(type) {
	case *TCPAddr:
		proto = "tcp"
		ip = a.IP
		port = a.Port
	case *UDPAddr:
		proto = "udp"
		ip = a.IP
		port = a.Port
	default:
		err = UnknownNetworkError(net)
		return
	}

	clone, dest, err := queryCS1(proto, ip, port)
	if err != nil {
		return
	}
	f, err := os.OpenFile(clone, os.O_RDWR, 0)
	if err != nil {
		return
	}
	var buf [16]byte
	n, err := f.Read(buf[:])
	if err != nil {
		return
	}
	return f, dest, proto, string(buf[:n]), nil
}

func dialPlan9(net string, laddr, raddr Addr) (c *plan9Conn, err error) {
	f, dest, proto, name, err := startPlan9(net, raddr)
	if err != nil {
		return
	}
	_, err = f.WriteString("connect " + dest)
	if err != nil {
		return
	}
	laddr, err = readPlan9Addr(proto, "/net/"+proto+"/"+name+"/local")
	if err != nil {
		return
	}
	raddr, err = readPlan9Addr(proto, "/net/"+proto+"/"+name+"/remote")
	if err != nil {
		return
	}
	return newPlan9Conn(proto, name, f, laddr, raddr), nil
}

type plan9Listener struct {
	proto, name, dir string
	ctl              *os.File
	laddr            Addr
}

func listenPlan9(net string, laddr Addr) (l *plan9Listener, err error) {
	f, dest, proto, name, err := startPlan9(net, laddr)
	if err != nil {
		return
	}
	_, err = f.WriteString("announce " + dest)
	if err != nil {
		return
	}
	laddr, err = readPlan9Addr(proto, "/net/"+proto+"/"+name+"/local")
	if err != nil {
		return
	}
	l = new(plan9Listener)
	l.proto = proto
	l.name = name
	l.dir = "/net/" + proto + "/" + name
	l.ctl = f
	l.laddr = laddr
	return l, nil
}

func (l *plan9Listener) plan9Conn() *plan9Conn {
	return newPlan9Conn(l.proto, l.name, l.ctl, l.laddr, nil)
}

func (l *plan9Listener) acceptPlan9() (c *plan9Conn, err error) {
	f, err := os.Open(l.dir + "/listen")
	if err != nil {
		return
	}
	var buf [16]byte
	n, err := f.Read(buf[:])
	if err != nil {
		return
	}
	name := string(buf[:n])
	laddr, err := readPlan9Addr(l.proto, l.dir+"/local")
	if err != nil {
		return
	}
	raddr, err := readPlan9Addr(l.proto, l.dir+"/remote")
	if err != nil {
		return
	}
	return newPlan9Conn(l.proto, name, f, laddr, raddr), nil
}

func (l *plan9Listener) Accept() (c Conn, err error) {
	c1, err := l.acceptPlan9()
	if err != nil {
		return
	}
	return c1, nil
}

func (l *plan9Listener) Close() error {
	if l == nil || l.ctl == nil {
		return os.EINVAL
	}
	return l.ctl.Close()
}

func (l *plan9Listener) Addr() Addr { return l.laddr }

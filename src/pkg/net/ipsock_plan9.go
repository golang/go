// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Internet protocol family sockets for Plan 9

package net

import (
	"errors"
	"os"
	"syscall"
)

// /sys/include/ape/sys/socket.h:/SOMAXCONN
var listenerBacklog = 5

// probeIPv6Stack returns two boolean values.  If the first boolean
// value is true, kernel supports basic IPv6 functionality.  If the
// second boolean value is true, kernel supports IPv6 IPv4-mapping.
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
	defer f.Close()
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
		addr = &TCPAddr{IP: ip, Port: port}
	case "udp":
		addr = &UDPAddr{IP: ip, Port: port}
	default:
		return nil, errors.New("unknown protocol " + proto)
	}
	return addr, nil
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
		f.Close()
		return
	}
	return f, dest, proto, string(buf[:n]), nil
}

func netErr(e error) {
	oe, ok := e.(*OpError)
	if !ok {
		return
	}
	if pe, ok := oe.Err.(*os.PathError); ok {
		if _, ok = pe.Err.(syscall.ErrorString); ok {
			oe.Err = pe.Err
		}
	}
}

func dialPlan9(net string, laddr, raddr Addr) (fd *netFD, err error) {
	defer func() { netErr(err) }()
	f, dest, proto, name, err := startPlan9(net, raddr)
	if err != nil {
		return nil, &OpError{"dial", net, raddr, err}
	}
	_, err = f.WriteString("connect " + dest)
	if err != nil {
		f.Close()
		return nil, &OpError{"dial", f.Name(), raddr, err}
	}
	data, err := os.OpenFile("/net/"+proto+"/"+name+"/data", os.O_RDWR, 0)
	if err != nil {
		f.Close()
		return nil, &OpError{"dial", net, raddr, err}
	}
	laddr, err = readPlan9Addr(proto, "/net/"+proto+"/"+name+"/local")
	if err != nil {
		data.Close()
		f.Close()
		return nil, &OpError{"dial", proto, raddr, err}
	}
	return newFD(proto, name, f, data, laddr, raddr), nil
}

func listenPlan9(net string, laddr Addr) (fd *netFD, err error) {
	defer func() { netErr(err) }()
	f, dest, proto, name, err := startPlan9(net, laddr)
	if err != nil {
		return nil, &OpError{"listen", net, laddr, err}
	}
	_, err = f.WriteString("announce " + dest)
	if err != nil {
		f.Close()
		return nil, &OpError{"announce", proto, laddr, err}
	}
	laddr, err = readPlan9Addr(proto, "/net/"+proto+"/"+name+"/local")
	if err != nil {
		f.Close()
		return nil, &OpError{Op: "listen", Net: net, Err: err}
	}
	return newFD(proto, name, f, nil, laddr, nil), nil
}

func (l *netFD) netFD() *netFD {
	return newFD(l.proto, l.name, l.ctl, l.data, l.laddr, l.raddr)
}

func (l *netFD) acceptPlan9() (fd *netFD, err error) {
	defer func() { netErr(err) }()
	f, err := os.Open(l.dir + "/listen")
	if err != nil {
		return nil, &OpError{"accept", l.dir + "/listen", l.laddr, err}
	}
	var buf [16]byte
	n, err := f.Read(buf[:])
	if err != nil {
		f.Close()
		return nil, &OpError{"accept", l.dir + "/listen", l.laddr, err}
	}
	name := string(buf[:n])
	data, err := os.OpenFile("/net/"+l.proto+"/"+name+"/data", os.O_RDWR, 0)
	if err != nil {
		f.Close()
		return nil, &OpError{"accept", l.proto, l.laddr, err}
	}
	raddr, err := readPlan9Addr(l.proto, "/net/"+l.proto+"/"+name+"/remote")
	if err != nil {
		data.Close()
		f.Close()
		return nil, &OpError{"accept", l.proto, l.laddr, err}
	}
	return newFD(l.proto, name, f, data, l.laddr, raddr), nil
}

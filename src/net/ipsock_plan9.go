// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Internet protocol family sockets for Plan 9

package net

import (
	"context"
	"os"
	"syscall"
)

func probe(filename, query string) bool {
	var file *file
	var err error
	if file, err = open(filename); err != nil {
		return false
	}

	r := false
	for line, ok := file.readLine(); ok && !r; line, ok = file.readLine() {
		f := getFields(line)
		if len(f) < 3 {
			continue
		}
		for i := 0; i < len(f); i++ {
			if query == f[i] {
				r = true
				break
			}
		}
	}
	file.close()
	return r
}

func probeIPv4Stack() bool {
	return probe(netdir+"/iproute", "4i")
}

// probeIPv6Stack returns two boolean values. If the first boolean
// value is true, kernel supports basic IPv6 functionality. If the
// second boolean value is true, kernel supports IPv6 IPv4-mapping.
func probeIPv6Stack() (supportsIPv6, supportsIPv4map bool) {
	// Plan 9 uses IPv6 natively, see ip(3).
	r := probe(netdir+"/iproute", "6i")
	v := false
	if r {
		v = probe(netdir+"/iproute", "4i")
	}
	return r, v
}

// parsePlan9Addr parses address of the form [ip!]port (e.g. 127.0.0.1!80).
func parsePlan9Addr(s string) (ip IP, iport int, err error) {
	addr := IPv4zero // address contains port only
	i := byteIndex(s, '!')
	if i >= 0 {
		addr = ParseIP(s[:i])
		if addr == nil {
			return nil, 0, &ParseError{Type: "IP address", Text: s}
		}
	}
	p, _, ok := dtoi(s[i+1:], 0)
	if !ok {
		return nil, 0, &ParseError{Type: "port", Text: s}
	}
	if p < 0 || p > 0xFFFF {
		return nil, 0, &AddrError{Err: "invalid port", Addr: string(p)}
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
		return nil, UnknownNetworkError(proto)
	}
	return addr, nil
}

func startPlan9(ctx context.Context, net string, addr Addr) (ctl *os.File, dest, proto, name string, err error) {
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

	clone, dest, err := queryCS1(ctx, proto, ip, port)
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

func fixErr(err error) {
	oe, ok := err.(*OpError)
	if !ok {
		return
	}
	nonNilInterface := func(a Addr) bool {
		switch a := a.(type) {
		case *TCPAddr:
			return a == nil
		case *UDPAddr:
			return a == nil
		case *IPAddr:
			return a == nil
		default:
			return false
		}
	}
	if nonNilInterface(oe.Source) {
		oe.Source = nil
	}
	if nonNilInterface(oe.Addr) {
		oe.Addr = nil
	}
	if pe, ok := oe.Err.(*os.PathError); ok {
		if _, ok = pe.Err.(syscall.ErrorString); ok {
			oe.Err = pe.Err
		}
	}
}

func dialPlan9(ctx context.Context, net string, laddr, raddr Addr) (fd *netFD, err error) {
	defer func() { fixErr(err) }()
	type res struct {
		fd  *netFD
		err error
	}
	resc := make(chan res)
	go func() {
		testHookDialChannel()
		fd, err := dialPlan9Blocking(ctx, net, laddr, raddr)
		select {
		case resc <- res{fd, err}:
		case <-ctx.Done():
			if fd != nil {
				fd.Close()
			}
		}
	}()
	select {
	case res := <-resc:
		return res.fd, res.err
	case <-ctx.Done():
		return nil, mapErr(ctx.Err())
	}
}

func dialPlan9Blocking(ctx context.Context, net string, laddr, raddr Addr) (fd *netFD, err error) {
	f, dest, proto, name, err := startPlan9(ctx, net, raddr)
	if err != nil {
		return nil, err
	}
	_, err = f.WriteString("connect " + dest)
	if err != nil {
		f.Close()
		return nil, err
	}
	data, err := os.OpenFile(netdir+"/"+proto+"/"+name+"/data", os.O_RDWR, 0)
	if err != nil {
		f.Close()
		return nil, err
	}
	laddr, err = readPlan9Addr(proto, netdir+"/"+proto+"/"+name+"/local")
	if err != nil {
		data.Close()
		f.Close()
		return nil, err
	}
	return newFD(proto, name, f, data, laddr, raddr)
}

func listenPlan9(ctx context.Context, net string, laddr Addr) (fd *netFD, err error) {
	defer func() { fixErr(err) }()
	f, dest, proto, name, err := startPlan9(ctx, net, laddr)
	if err != nil {
		return nil, err
	}
	_, err = f.WriteString("announce " + dest)
	if err != nil {
		f.Close()
		return nil, err
	}
	laddr, err = readPlan9Addr(proto, netdir+"/"+proto+"/"+name+"/local")
	if err != nil {
		f.Close()
		return nil, err
	}
	return newFD(proto, name, f, nil, laddr, nil)
}

func (fd *netFD) netFD() (*netFD, error) {
	return newFD(fd.net, fd.n, fd.ctl, fd.data, fd.laddr, fd.raddr)
}

func (fd *netFD) acceptPlan9() (nfd *netFD, err error) {
	defer func() { fixErr(err) }()
	if err := fd.readLock(); err != nil {
		return nil, err
	}
	defer fd.readUnlock()
	f, err := os.Open(fd.dir + "/listen")
	if err != nil {
		return nil, err
	}
	var buf [16]byte
	n, err := f.Read(buf[:])
	if err != nil {
		f.Close()
		return nil, err
	}
	name := string(buf[:n])
	data, err := os.OpenFile(netdir+"/"+fd.net+"/"+name+"/data", os.O_RDWR, 0)
	if err != nil {
		f.Close()
		return nil, err
	}
	raddr, err := readPlan9Addr(fd.net, netdir+"/"+fd.net+"/"+name+"/remote")
	if err != nil {
		data.Close()
		f.Close()
		return nil, err
	}
	return newFD(fd.net, name, f, data, fd.laddr, raddr)
}

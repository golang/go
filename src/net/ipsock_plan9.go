// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"internal/bytealg"
	"internal/itoa"
	"io/fs"
	"os"
	"syscall"
)

// probe probes IPv4, IPv6 and IPv4-mapped IPv6 communication
// capabilities.
//
// Plan 9 uses IPv6 natively, see ip(3).
func (p *ipStackCapabilities) probe() {
	p.ipv4Enabled = probe(netdir+"/iproute", "4i")
	p.ipv6Enabled = probe(netdir+"/iproute", "6i")
	if p.ipv4Enabled && p.ipv6Enabled {
		p.ipv4MappedIPv6Enabled = true
	}
}

func probe(filename, query string) bool {
	var file *file
	var err error
	if file, err = open(filename); err != nil {
		return false
	}
	defer file.close()

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
	return r
}

// parsePlan9Addr parses address of the form [ip!]port (e.g. 127.0.0.1!80).
func parsePlan9Addr(s string) (ip IP, iport int, err error) {
	addr := IPv4zero // address contains port only
	i := bytealg.IndexByteString(s, '!')
	if i >= 0 {
		addr = ParseIP(s[:i])
		if addr == nil {
			return nil, 0, &ParseError{Type: "IP address", Text: s}
		}
	}
	p, plen, ok := dtoi(s[i+1:])
	if !ok {
		return nil, 0, &ParseError{Type: "port", Text: s}
	}
	if p < 0 || p > 0xFFFF {
		return nil, 0, &AddrError{Err: "invalid port", Addr: s[i+1 : i+1+plen]}
	}
	return addr, p, nil
}

func readPlan9Addr(net, filename string) (addr Addr, err error) {
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
	switch net {
	case "tcp4", "udp4":
		if ip.Equal(IPv6zero) {
			ip = ip[:IPv4len]
		}
	}
	switch net {
	case "tcp", "tcp4", "tcp6":
		addr = &TCPAddr{IP: ip, Port: port}
	case "udp", "udp4", "udp6":
		addr = &UDPAddr{IP: ip, Port: port}
	default:
		return nil, UnknownNetworkError(net)
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

	if port > 65535 {
		err = InvalidAddrError("port should be < 65536")
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
	if pe, ok := oe.Err.(*fs.PathError); ok {
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
	if isWildcard(raddr) {
		raddr = toLocal(raddr, net)
	}
	f, dest, proto, name, err := startPlan9(ctx, net, raddr)
	if err != nil {
		return nil, err
	}
	if la := plan9LocalAddr(laddr); la == "" {
		err = hangupCtlWrite(ctx, proto, f, "connect "+dest)
	} else {
		err = hangupCtlWrite(ctx, proto, f, "connect "+dest+" "+la)
	}
	if err != nil {
		f.Close()
		return nil, err
	}
	data, err := os.OpenFile(netdir+"/"+proto+"/"+name+"/data", os.O_RDWR, 0)
	if err != nil {
		f.Close()
		return nil, err
	}
	laddr, err = readPlan9Addr(net, netdir+"/"+proto+"/"+name+"/local")
	if err != nil {
		data.Close()
		f.Close()
		return nil, err
	}
	return newFD(proto, name, nil, f, data, laddr, raddr)
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
		return nil, &OpError{Op: "announce", Net: net, Source: laddr, Addr: nil, Err: err}
	}
	laddr, err = readPlan9Addr(net, netdir+"/"+proto+"/"+name+"/local")
	if err != nil {
		f.Close()
		return nil, err
	}
	return newFD(proto, name, nil, f, nil, laddr, nil)
}

func (fd *netFD) netFD() (*netFD, error) {
	return newFD(fd.net, fd.n, fd.listen, fd.ctl, fd.data, fd.laddr, fd.raddr)
}

func (fd *netFD) acceptPlan9() (nfd *netFD, err error) {
	defer func() { fixErr(err) }()
	if err := fd.pfd.ReadLock(); err != nil {
		return nil, err
	}
	defer fd.pfd.ReadUnlock()
	listen, err := os.Open(fd.dir + "/listen")
	if err != nil {
		return nil, err
	}
	var buf [16]byte
	n, err := listen.Read(buf[:])
	if err != nil {
		listen.Close()
		return nil, err
	}
	name := string(buf[:n])
	ctl, err := os.OpenFile(netdir+"/"+fd.net+"/"+name+"/ctl", os.O_RDWR, 0)
	if err != nil {
		listen.Close()
		return nil, err
	}
	data, err := os.OpenFile(netdir+"/"+fd.net+"/"+name+"/data", os.O_RDWR, 0)
	if err != nil {
		listen.Close()
		ctl.Close()
		return nil, err
	}
	raddr, err := readPlan9Addr(fd.net, netdir+"/"+fd.net+"/"+name+"/remote")
	if err != nil {
		listen.Close()
		ctl.Close()
		data.Close()
		return nil, err
	}
	return newFD(fd.net, name, listen, ctl, data, fd.laddr, raddr)
}

func isWildcard(a Addr) bool {
	var wildcard bool
	switch a := a.(type) {
	case *TCPAddr:
		wildcard = a.isWildcard()
	case *UDPAddr:
		wildcard = a.isWildcard()
	case *IPAddr:
		wildcard = a.isWildcard()
	}
	return wildcard
}

func toLocal(a Addr, net string) Addr {
	switch a := a.(type) {
	case *TCPAddr:
		a.IP = loopbackIP(net)
	case *UDPAddr:
		a.IP = loopbackIP(net)
	case *IPAddr:
		a.IP = loopbackIP(net)
	}
	return a
}

// plan9LocalAddr returns a Plan 9 local address string.
// See setladdrport at https://9p.io/sources/plan9/sys/src/9/ip/devip.c.
func plan9LocalAddr(addr Addr) string {
	var ip IP
	port := 0
	switch a := addr.(type) {
	case *TCPAddr:
		if a != nil {
			ip = a.IP
			port = a.Port
		}
	case *UDPAddr:
		if a != nil {
			ip = a.IP
			port = a.Port
		}
	}
	if len(ip) == 0 || ip.IsUnspecified() {
		if port == 0 {
			return ""
		}
		return itoa.Itoa(port)
	}
	return ip.String() + "!" + itoa.Itoa(port)
}

func hangupCtlWrite(ctx context.Context, proto string, ctl *os.File, msg string) error {
	if proto != "tcp" {
		_, err := ctl.WriteString(msg)
		return err
	}
	written := make(chan struct{})
	errc := make(chan error)
	go func() {
		select {
		case <-ctx.Done():
			ctl.WriteString("hangup")
			errc <- mapErr(ctx.Err())
		case <-written:
			errc <- nil
		}
	}()
	_, err := ctl.WriteString(msg)
	close(written)
	if e := <-errc; err == nil && e != nil { // we hung up
		return e
	}
	return err
}

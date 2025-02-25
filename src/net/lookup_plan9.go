// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"context"
	"errors"
	"internal/bytealg"
	"internal/itoa"
	"internal/stringslite"
	"io"
	"os"
)

// cgoAvailable set to true to indicate that the cgo resolver
// is available on Plan 9. Note that on Plan 9 the cgo resolver
// does not actually use cgo.
const cgoAvailable = true

func query(ctx context.Context, filename, query string, bufSize int) (addrs []string, err error) {
	queryAddrs := func() (addrs []string, err error) {
		file, err := os.OpenFile(filename, os.O_RDWR, 0)
		if err != nil {
			return nil, err
		}
		defer file.Close()

		_, err = file.Seek(0, io.SeekStart)
		if err != nil {
			return nil, err
		}
		_, err = file.WriteString(query)
		if err != nil {
			return nil, err
		}
		_, err = file.Seek(0, io.SeekStart)
		if err != nil {
			return nil, err
		}
		buf := make([]byte, bufSize)
		for {
			n, _ := file.Read(buf)
			if n <= 0 {
				break
			}
			addrs = append(addrs, string(buf[:n]))
		}
		return addrs, nil
	}

	type ret struct {
		addrs []string
		err   error
	}

	ch := make(chan ret, 1)
	go func() {
		addrs, err := queryAddrs()
		ch <- ret{addrs: addrs, err: err}
	}()

	select {
	case r := <-ch:
		return r.addrs, r.err
	case <-ctx.Done():
		return nil, mapErr(err)
	}
}

func queryCS(ctx context.Context, net, host, service string) (res []string, err error) {
	switch net {
	case "tcp4", "tcp6":
		net = "tcp"
	case "udp4", "udp6":
		net = "udp"
	}
	if host == "" {
		host = "*"
	}
	return query(ctx, netdir+"/cs", net+"!"+host+"!"+service, 128)
}

func queryCS1(ctx context.Context, net string, ip IP, port int) (clone, dest string, err error) {
	ips := "*"
	if len(ip) != 0 && !ip.IsUnspecified() {
		ips = ip.String()
	}
	lines, err := queryCS(ctx, net, ips, itoa.Itoa(port))
	if err != nil {
		return
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return "", "", errors.New("bad response from ndb/cs")
	}
	clone, dest = f[0], f[1]
	return
}

func queryDNS(ctx context.Context, addr string, typ string) (res []string, err error) {
	return query(ctx, netdir+"/dns", addr+" "+typ, 1024)
}

func handlePlan9DNSError(err error, name string) error {
	if stringslite.HasSuffix(err.Error(), "dns: name does not exist") ||
		stringslite.HasSuffix(err.Error(), "dns: resource does not exist; negrcode 0") ||
		stringslite.HasSuffix(err.Error(), "dns: resource does not exist; negrcode") ||
		stringslite.HasSuffix(err.Error(), "dns failure") {
		err = errNoSuchHost
	}
	return newDNSError(err, name, "")
}

// toLower returns a lower-case version of in. Restricting us to
// ASCII is sufficient to handle the IP protocol names and allow
// us to not depend on the strings and unicode packages.
func toLower(in string) string {
	for _, c := range in {
		if 'A' <= c && c <= 'Z' {
			// Has upper case; need to fix.
			out := []byte(in)
			for i := 0; i < len(in); i++ {
				c := in[i]
				if 'A' <= c && c <= 'Z' {
					c += 'a' - 'A'
				}
				out[i] = c
			}
			return string(out)
		}
	}
	return in
}

// lookupProtocol looks up IP protocol name and returns
// the corresponding protocol number.
func lookupProtocol(ctx context.Context, name string) (proto int, err error) {
	lines, err := query(ctx, netdir+"/cs", "!protocol="+toLower(name), 128)
	if err != nil {
		return 0, newDNSError(err, name, "")
	}
	if len(lines) == 0 {
		return 0, UnknownNetworkError(name)
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return 0, UnknownNetworkError(name)
	}
	s := f[1]
	if n, _, ok := dtoi(s[bytealg.IndexByteString(s, '=')+1:]); ok {
		return n, nil
	}
	return 0, UnknownNetworkError(name)
}

func (*Resolver) lookupHost(ctx context.Context, host string) (addrs []string, err error) {
	// Use netdir/cs instead of netdir/dns because cs knows about
	// host names in local network (e.g. from /lib/ndb/local)
	lines, err := queryCS(ctx, "net", host, "1")
	if err != nil {
		return nil, handlePlan9DNSError(err, host)
	}
loop:
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 2 {
			continue
		}
		addr := f[1]
		if i := bytealg.IndexByteString(addr, '!'); i >= 0 {
			addr = addr[:i] // remove port
		}
		if ParseIP(addr) == nil {
			continue
		}
		// only return unique addresses
		for _, a := range addrs {
			if a == addr {
				continue loop
			}
		}
		addrs = append(addrs, addr)
	}
	return
}

func (r *Resolver) lookupIP(ctx context.Context, network, host string) (addrs []IPAddr, err error) {
	if order, conf := systemConf().hostLookupOrder(r, host); order != hostLookupCgo {
		return r.goLookupIP(ctx, network, host, order, conf)
	}

	lits, err := r.lookupHost(ctx, host)
	if err != nil {
		return
	}
	for _, lit := range lits {
		host, zone := splitHostZone(lit)
		if ip := ParseIP(host); ip != nil {
			addr := IPAddr{IP: ip, Zone: zone}
			addrs = append(addrs, addr)
		}
	}
	return
}

func (r *Resolver) lookupPort(ctx context.Context, network, service string) (port int, err error) {
	switch network {
	case "ip": // no hints
		if p, err := r.lookupPortWithNetwork(ctx, "tcp", "ip", service); err == nil {
			return p, nil
		}
		return r.lookupPortWithNetwork(ctx, "udp", "ip", service)
	case "tcp", "tcp4", "tcp6":
		return r.lookupPortWithNetwork(ctx, "tcp", "tcp", service)
	case "udp", "udp4", "udp6":
		return r.lookupPortWithNetwork(ctx, "udp", "udp", service)
	default:
		return 0, &DNSError{Err: "unknown network", Name: network + "/" + service}
	}
}

func (*Resolver) lookupPortWithNetwork(ctx context.Context, network, errNetwork, service string) (port int, err error) {
	lines, err := queryCS(ctx, network, "127.0.0.1", toLower(service))
	if err != nil {
		if stringslite.HasSuffix(err.Error(), "can't translate service") {
			return 0, newDNSError(errUnknownPort, errNetwork+"/"+service, "")
		}
		return 0, newDNSError(err, errNetwork+"/"+service, "")
	}
	if len(lines) == 0 {
		return 0, newDNSError(errUnknownPort, errNetwork+"/"+service, "")
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return 0, newDNSError(errUnknownPort, errNetwork+"/"+service, "")
	}
	s := f[1]
	if i := bytealg.IndexByteString(s, '!'); i >= 0 {
		s = s[i+1:] // remove address
	}
	if n, _, ok := dtoi(s); ok {
		return n, nil
	}
	return 0, newDNSError(errUnknownPort, errNetwork+"/"+service, "")
}

func (r *Resolver) lookupCNAME(ctx context.Context, name string) (cname string, err error) {
	if order, conf := systemConf().hostLookupOrder(r, name); order != hostLookupCgo {
		return r.goLookupCNAME(ctx, name, order, conf)
	}

	lines, err := queryDNS(ctx, name, "cname")
	if err != nil {
		if stringslite.HasSuffix(err.Error(), "dns failure") ||
			stringslite.HasSuffix(err.Error(), "resource does not exist; negrcode 0") ||
			stringslite.HasSuffix(err.Error(), "resource does not exist; negrcode") {
			return absDomainName(name), nil
		}
		return "", handlePlan9DNSError(err, cname)
	}
	if len(lines) > 0 {
		if f := getFields(lines[0]); len(f) >= 3 {
			return f[2] + ".", nil
		}
	}
	return "", &DNSError{Err: "bad response from ndb/dns", Name: name}
}

func (r *Resolver) lookupSRV(ctx context.Context, service, proto, name string) (cname string, addrs []*SRV, err error) {
	if systemConf().mustUseGoResolver(r) {
		return r.goLookupSRV(ctx, service, proto, name)
	}
	var target string
	if service == "" && proto == "" {
		target = name
	} else {
		target = "_" + service + "._" + proto + "." + name
	}
	lines, err := queryDNS(ctx, target, "srv")
	if err != nil {
		return "", nil, handlePlan9DNSError(err, name)
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 6 {
			continue
		}
		port, _, portOk := dtoi(f[4])
		priority, _, priorityOk := dtoi(f[3])
		weight, _, weightOk := dtoi(f[2])
		if !(portOk && priorityOk && weightOk) {
			continue
		}
		addrs = append(addrs, &SRV{absDomainName(f[5]), uint16(port), uint16(priority), uint16(weight)})
		cname = absDomainName(f[0])
	}
	byPriorityWeight(addrs).sort()
	return
}

func (r *Resolver) lookupMX(ctx context.Context, name string) (mx []*MX, err error) {
	if systemConf().mustUseGoResolver(r) {
		return r.goLookupMX(ctx, name)
	}
	lines, err := queryDNS(ctx, name, "mx")
	if err != nil {
		return nil, handlePlan9DNSError(err, name)
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 4 {
			continue
		}
		if pref, _, ok := dtoi(f[2]); ok {
			mx = append(mx, &MX{absDomainName(f[3]), uint16(pref)})
		}
	}
	byPref(mx).sort()
	return
}

func (r *Resolver) lookupNS(ctx context.Context, name string) (ns []*NS, err error) {
	if systemConf().mustUseGoResolver(r) {
		return r.goLookupNS(ctx, name)
	}
	lines, err := queryDNS(ctx, name, "ns")
	if err != nil {
		return nil, handlePlan9DNSError(err, name)
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 3 {
			continue
		}
		ns = append(ns, &NS{absDomainName(f[2])})
	}
	return
}

func (r *Resolver) lookupTXT(ctx context.Context, name string) (txt []string, err error) {
	if systemConf().mustUseGoResolver(r) {
		return r.goLookupTXT(ctx, name)
	}
	lines, err := queryDNS(ctx, name, "txt")
	if err != nil {
		return nil, handlePlan9DNSError(err, name)
	}
	for _, line := range lines {
		if i := bytealg.IndexByteString(line, '\t'); i >= 0 {
			txt = append(txt, line[i+1:])
		}
	}
	return
}

func (r *Resolver) lookupAddr(ctx context.Context, addr string) (name []string, err error) {
	if order, conf := systemConf().addrLookupOrder(r, addr); order != hostLookupCgo {
		return r.goLookupPTR(ctx, addr, order, conf)
	}
	arpa, err := reverseaddr(addr)
	if err != nil {
		return
	}
	lines, err := queryDNS(ctx, arpa, "ptr")
	if err != nil {
		return nil, handlePlan9DNSError(err, addr)
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 3 {
			continue
		}
		name = append(name, absDomainName(f[2]))
	}
	return
}

// concurrentThreadsLimit returns the number of threads we permit to
// run concurrently doing DNS lookups.
func concurrentThreadsLimit() int {
	return 500
}

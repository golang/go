// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
)

func query(filename, query string, bufSize int) (res []string, err os.Error) {
	file, err := os.OpenFile(filename, os.O_RDWR, 0)
	if err != nil {
		return
	}
	defer file.Close()

	_, err = file.WriteString(query)
	if err != nil {
		return
	}
	_, err = file.Seek(0, 0)
	if err != nil {
		return
	}
	buf := make([]byte, bufSize)
	for {
		n, _ := file.Read(buf)
		if n <= 0 {
			break
		}
		res = append(res, string(buf[:n]))
	}
	return
}

func queryCS(net, host, service string) (res []string, err os.Error) {
	switch net {
	case "tcp4", "tcp6":
		net = "tcp"
	case "udp4", "udp6":
		net = "udp"
	}
	if host == "" {
		host = "*"
	}
	return query("/net/cs", net+"!"+host+"!"+service, 128)
}

func queryCS1(net string, ip IP, port int) (clone, dest string, err os.Error) {
	ips := "*"
	if !ip.IsUnspecified() {
		ips = ip.String()
	}
	lines, err := queryCS(net, ips, itoa(port))
	if err != nil {
		return
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return "", "", os.NewError("net: bad response from ndb/cs")
	}
	clone, dest = f[0], f[1]
	return
}

func queryDNS(addr string, typ string) (res []string, err os.Error) {
	return query("/net/dns", addr+" "+typ, 1024)
}

// LookupHost looks up the given host using the local resolver.
// It returns an array of that host's addresses.
func LookupHost(host string) (addrs []string, err os.Error) {
	// Use /net/cs insead of /net/dns because cs knows about
	// host names in local network (e.g. from /lib/ndb/local)
	lines, err := queryCS("tcp", host, "1")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 2 {
			continue
		}
		addr := f[1]
		if i := byteIndex(addr, '!'); i >= 0 {
			addr = addr[:i] // remove port
		}
		if ParseIP(addr) == nil {
			continue
		}
		addrs = append(addrs, addr)
	}
	return
}

// LookupIP looks up host using the local resolver.
// It returns an array of that host's IPv4 and IPv6 addresses.
func LookupIP(host string) (ips []IP, err os.Error) {
	addrs, err := LookupHost(host)
	if err != nil {
		return
	}
	for _, addr := range addrs {
		if ip := ParseIP(addr); ip != nil {
			ips = append(ips, ip)
		}
	}
	return
}

// LookupPort looks up the port for the given network and service.
func LookupPort(network, service string) (port int, err os.Error) {
	switch network {
	case "tcp4", "tcp6":
		network = "tcp"
	case "udp4", "udp6":
		network = "udp"
	}
	lines, err := queryCS(network, "127.0.0.1", service)
	if err != nil {
		return
	}
	unknownPortError := &AddrError{"unknown port", network + "/" + service}
	if len(lines) == 0 {
		return 0, unknownPortError
	}
	f := getFields(lines[0])
	if len(f) < 2 {
		return 0, unknownPortError
	}
	s := f[1]
	if i := byteIndex(s, '!'); i >= 0 {
		s = s[i+1:] // remove address
	}
	if n, _, ok := dtoi(s, 0); ok {
		return n, nil
	}
	return 0, unknownPortError
}

// LookupCNAME returns the canonical DNS host for the given name.
// Callers that do not care about the canonical name can call
// LookupHost or LookupIP directly; both take care of resolving
// the canonical name as part of the lookup.
func LookupCNAME(name string) (cname string, err os.Error) {
	lines, err := queryDNS(name, "cname")
	if err != nil {
		return
	}
	if len(lines) > 0 {
		if f := getFields(lines[0]); len(f) >= 3 {
			return f[2] + ".", nil
		}
	}
	return "", os.NewError("net: bad response from ndb/dns")
}

// LookupSRV tries to resolve an SRV query of the given service,
// protocol, and domain name.  The proto is "tcp" or "udp".
// The returned records are sorted by priority and randomized
// by weight within a priority.
//
// LookupSRV constructs the DNS name to look up following RFC 2782.
// That is, it looks up _service._proto.name.  To accommodate services
// publishing SRV records under non-standard names, if both service
// and proto are empty strings, LookupSRV looks up name directly.
func LookupSRV(service, proto, name string) (cname string, addrs []*SRV, err os.Error) {
	var target string
	if service == "" && proto == "" {
		target = name
	} else {
		target = "_" + service + "._" + proto + "." + name
	}
	lines, err := queryDNS(target, "srv")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 6 {
			continue
		}
		port, _, portOk := dtoi(f[2], 0)
		priority, _, priorityOk := dtoi(f[3], 0)
		weight, _, weightOk := dtoi(f[4], 0)
		if !(portOk && priorityOk && weightOk) {
			continue
		}
		addrs = append(addrs, &SRV{f[5], uint16(port), uint16(priority), uint16(weight)})
		cname = f[0]
	}
	byPriorityWeight(addrs).sort()
	return
}

// LookupMX returns the DNS MX records for the given domain name sorted by preference.
func LookupMX(name string) (mx []*MX, err os.Error) {
	lines, err := queryDNS(name, "mx")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 4 {
			continue
		}
		if pref, _, ok := dtoi(f[2], 0); ok {
			mx = append(mx, &MX{f[3], uint16(pref)})
		}
	}
	byPref(mx).sort()
	return
}

// LookupTXT returns the DNS TXT records for the given domain name.
func LookupTXT(name string) (txt []string, err os.Error) {
	return nil, os.NewError("net.LookupTXT is not implemented on Plan 9")
}

// LookupAddr performs a reverse lookup for the given address, returning a list
// of names mapping to that address.
func LookupAddr(addr string) (name []string, err os.Error) {
	arpa, err := reverseaddr(addr)
	if err != nil {
		return
	}
	lines, err := queryDNS(arpa, "ptr")
	if err != nil {
		return
	}
	for _, line := range lines {
		f := getFields(line)
		if len(f) < 3 {
			continue
		}
		name = append(name, f[2])
	}
	return
}

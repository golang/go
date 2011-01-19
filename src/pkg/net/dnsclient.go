// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// DNS client: see RFC 1035.
// Has to be linked into package net for Dial.

// TODO(rsc):
//	Check periodically whether /etc/resolv.conf has changed.
//	Could potentially handle many outstanding lookups faster.
//	Could have a small cache.
//	Random UDP source port (net.Dial should do that for us).
//	Random request IDs.

package net

import (
	"bytes"
	"fmt"
	"os"
	"rand"
	"sync"
	"time"
)

// DNSError represents a DNS lookup error.
type DNSError struct {
	Error     string // description of the error
	Name      string // name looked for
	Server    string // server used
	IsTimeout bool
}

func (e *DNSError) String() string {
	if e == nil {
		return "<nil>"
	}
	s := "lookup " + e.Name
	if e.Server != "" {
		s += " on " + e.Server
	}
	s += ": " + e.Error
	return s
}

func (e *DNSError) Timeout() bool   { return e.IsTimeout }
func (e *DNSError) Temporary() bool { return e.IsTimeout }

const noSuchHost = "no such host"

// Send a request on the connection and hope for a reply.
// Up to cfg.attempts attempts.
func exchange(cfg *dnsConfig, c Conn, name string, qtype uint16) (*dnsMsg, os.Error) {
	if len(name) >= 256 {
		return nil, &DNSError{Error: "name too long", Name: name}
	}
	out := new(dnsMsg)
	out.id = uint16(rand.Int()) ^ uint16(time.Nanoseconds())
	out.question = []dnsQuestion{
		{name, qtype, dnsClassINET},
	}
	out.recursion_desired = true
	msg, ok := out.Pack()
	if !ok {
		return nil, &DNSError{Error: "internal error - cannot pack message", Name: name}
	}

	for attempt := 0; attempt < cfg.attempts; attempt++ {
		n, err := c.Write(msg)
		if err != nil {
			return nil, err
		}

		c.SetReadTimeout(int64(cfg.timeout) * 1e9) // nanoseconds

		buf := make([]byte, 2000) // More than enough.
		n, err = c.Read(buf)
		if err != nil {
			if e, ok := err.(Error); ok && e.Timeout() {
				continue
			}
			return nil, err
		}
		buf = buf[0:n]
		in := new(dnsMsg)
		if !in.Unpack(buf) || in.id != out.id {
			continue
		}
		return in, nil
	}
	var server string
	if a := c.RemoteAddr(); a != nil {
		server = a.String()
	}
	return nil, &DNSError{Error: "no answer from server", Name: name, Server: server, IsTimeout: true}
}


// Find answer for name in dns message.
// On return, if err == nil, addrs != nil.
func answer(name, server string, dns *dnsMsg, qtype uint16) (addrs []dnsRR, err os.Error) {
	addrs = make([]dnsRR, 0, len(dns.answer))

	if dns.rcode == dnsRcodeNameError && dns.recursion_available {
		return nil, &DNSError{Error: noSuchHost, Name: name}
	}
	if dns.rcode != dnsRcodeSuccess {
		// None of the error codes make sense
		// for the query we sent.  If we didn't get
		// a name error and we didn't get success,
		// the server is behaving incorrectly.
		return nil, &DNSError{Error: "server misbehaving", Name: name, Server: server}
	}

	// Look for the name.
	// Presotto says it's okay to assume that servers listed in
	// /etc/resolv.conf are recursive resolvers.
	// We asked for recursion, so it should have included
	// all the answers we need in this one packet.
Cname:
	for cnameloop := 0; cnameloop < 10; cnameloop++ {
		addrs = addrs[0:0]
		for i := 0; i < len(dns.answer); i++ {
			rr := dns.answer[i]
			h := rr.Header()
			if h.Class == dnsClassINET && h.Name == name {
				switch h.Rrtype {
				case qtype:
					n := len(addrs)
					addrs = addrs[0 : n+1]
					addrs[n] = rr
				case dnsTypeCNAME:
					// redirect to cname
					name = rr.(*dnsRR_CNAME).Cname
					continue Cname
				}
			}
		}
		if len(addrs) == 0 {
			return nil, &DNSError{Error: noSuchHost, Name: name, Server: server}
		}
		return addrs, nil
	}

	return nil, &DNSError{Error: "too many redirects", Name: name, Server: server}
}

// Do a lookup for a single name, which must be rooted
// (otherwise answer will not find the answers).
func tryOneName(cfg *dnsConfig, name string, qtype uint16) (addrs []dnsRR, err os.Error) {
	if len(cfg.servers) == 0 {
		return nil, &DNSError{Error: "no DNS servers", Name: name}
	}
	for i := 0; i < len(cfg.servers); i++ {
		// Calling Dial here is scary -- we have to be sure
		// not to dial a name that will require a DNS lookup,
		// or Dial will call back here to translate it.
		// The DNS config parser has already checked that
		// all the cfg.servers[i] are IP addresses, which
		// Dial will use without a DNS lookup.
		server := cfg.servers[i] + ":53"
		c, cerr := Dial("udp", "", server)
		if cerr != nil {
			err = cerr
			continue
		}
		msg, merr := exchange(cfg, c, name, qtype)
		c.Close()
		if merr != nil {
			err = merr
			continue
		}
		addrs, err = answer(name, server, msg, qtype)
		if err == nil || err.(*DNSError).Error == noSuchHost {
			break
		}
	}
	return
}

func convertRR_A(records []dnsRR) []string {
	addrs := make([]string, len(records))
	for i := 0; i < len(records); i++ {
		rr := records[i]
		a := rr.(*dnsRR_A).A
		addrs[i] = IPv4(byte(a>>24), byte(a>>16), byte(a>>8), byte(a)).String()
	}
	return addrs
}

var cfg *dnsConfig
var dnserr os.Error

func loadConfig() { cfg, dnserr = dnsReadConfig() }

func isDomainName(s string) bool {
	// See RFC 1035, RFC 3696.
	if len(s) == 0 {
		return false
	}
	if len(s) > 255 {
		return false
	}
	if s[len(s)-1] != '.' { // simplify checking loop: make name end in dot
		s += "."
	}

	last := byte('.')
	ok := false // ok once we've seen a letter
	partlen := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		default:
			return false
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || c == '_':
			ok = true
			partlen++
		case '0' <= c && c <= '9':
			// fine
			partlen++
		case c == '-':
			// byte before dash cannot be dot
			if last == '.' {
				return false
			}
			partlen++
		case c == '.':
			// byte before dot cannot be dot, dash
			if last == '.' || last == '-' {
				return false
			}
			if partlen > 63 || partlen == 0 {
				return false
			}
			partlen = 0
		}
		last = c
	}

	return ok
}

var onceLoadConfig sync.Once

func lookup(name string, qtype uint16) (cname string, addrs []dnsRR, err os.Error) {
	if !isDomainName(name) {
		return name, nil, &DNSError{Error: "invalid domain name", Name: name}
	}
	onceLoadConfig.Do(loadConfig)
	if dnserr != nil || cfg == nil {
		err = dnserr
		return
	}
	// If name is rooted (trailing dot) or has enough dots,
	// try it by itself first.
	rooted := len(name) > 0 && name[len(name)-1] == '.'
	if rooted || count(name, '.') >= cfg.ndots {
		rname := name
		if !rooted {
			rname += "."
		}
		// Can try as ordinary name.
		addrs, err = tryOneName(cfg, rname, qtype)
		if err == nil {
			cname = rname
			return
		}
	}
	if rooted {
		return
	}

	// Otherwise, try suffixes.
	for i := 0; i < len(cfg.search); i++ {
		rname := name + "." + cfg.search[i]
		if rname[len(rname)-1] != '.' {
			rname += "."
		}
		addrs, err = tryOneName(cfg, rname, qtype)
		if err == nil {
			cname = rname
			return
		}
	}

	// Last ditch effort: try unsuffixed.
	rname := name
	if !rooted {
		rname += "."
	}
	addrs, err = tryOneName(cfg, rname, qtype)
	if err == nil {
		cname = rname
		return
	}
	return
}

// LookupHost looks for name using the local hosts file and DNS resolver.
// It returns the canonical name for the host and an array of that
// host's addresses.
func LookupHost(name string) (cname string, addrs []string, err os.Error) {
	onceLoadConfig.Do(loadConfig)
	if dnserr != nil || cfg == nil {
		err = dnserr
		return
	}
	// Use entries from /etc/hosts if they match.
	addrs = lookupStaticHost(name)
	if len(addrs) > 0 {
		cname = name
		return
	}
	var records []dnsRR
	cname, records, err = lookup(name, dnsTypeA)
	if err != nil {
		return
	}
	addrs = convertRR_A(records)
	return
}

type SRV struct {
	Target   string
	Port     uint16
	Priority uint16
	Weight   uint16
}

// LookupSRV tries to resolve an SRV query of the given service,
// protocol, and domain name, as specified in RFC 2782. In most cases
// the proto argument can be the same as the corresponding
// Addr.Network().
func LookupSRV(service, proto, name string) (cname string, addrs []*SRV, err os.Error) {
	target := "_" + service + "._" + proto + "." + name
	var records []dnsRR
	cname, records, err = lookup(target, dnsTypeSRV)
	if err != nil {
		return
	}
	addrs = make([]*SRV, len(records))
	for i := 0; i < len(records); i++ {
		r := records[i].(*dnsRR_SRV)
		addrs[i] = &SRV{r.Target, r.Port, r.Priority, r.Weight}
	}
	return
}

type MX struct {
	Host string
	Pref uint16
}

func LookupMX(name string) (entries []*MX, err os.Error) {
	var records []dnsRR
	_, records, err = lookup(name, dnsTypeMX)
	if err != nil {
		return
	}
	entries = make([]*MX, len(records))
	for i := range records {
		r := records[i].(*dnsRR_MX)
		entries[i] = &MX{r.Mx, r.Pref}
	}
	return
}

// reverseaddr returns the in-addr.arpa. or ip6.arpa. hostname of the IP
// address addr suitable for rDNS (PTR) record lookup or an error if it fails
// to parse the IP address.
func reverseaddr(addr string) (arpa string, err os.Error) {
	ip := ParseIP(addr)
	if ip == nil {
		return "", &DNSError{Error: "unrecognized address", Name: addr}
	}
	if ip.To4() != nil {
		return fmt.Sprintf("%d.%d.%d.%d.in-addr.arpa.", ip[15], ip[14], ip[13], ip[12]), nil
	}
	// Must be IPv6
	var buf bytes.Buffer
	// Add it, in reverse, to the buffer
	for i := len(ip) - 1; i >= 0; i-- {
		s := fmt.Sprintf("%02x", ip[i])
		buf.WriteByte(s[1])
		buf.WriteByte('.')
		buf.WriteByte(s[0])
		buf.WriteByte('.')
	}
	// Append "ip6.arpa." and return (buf already has the final .)
	return buf.String() + "ip6.arpa.", nil
}

// LookupAddr performs a reverse lookup for the given address, returning a list
// of names mapping to that address.
func LookupAddr(addr string) (name []string, err os.Error) {
	name = lookupStaticAddr(addr)
	if len(name) > 0 {
		return
	}
	var arpa string
	arpa, err = reverseaddr(addr)
	if err != nil {
		return
	}
	var records []dnsRR
	_, records, err = lookup(arpa, dnsTypePTR)
	if err != nil {
		return
	}
	name = make([]string, len(records))
	for i := range records {
		r := records[i].(*dnsRR_PTR)
		name[i] = r.Ptr
	}
	return
}

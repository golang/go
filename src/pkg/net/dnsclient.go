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
	"once"
	"os"
)

// DNSError represents a DNS lookup error.
type DNSError struct {
	Error  string // description of the error
	Name   string // name looked for
	Server string // server used
}

func (e *DNSError) String() string {
	s := "lookup " + e.Name
	if e.Server != "" {
		s += " on " + e.Server
	}
	s += ": " + e.Error
	return s
}

const noSuchHost = "no such host"

// Send a request on the connection and hope for a reply.
// Up to cfg.attempts attempts.
func _Exchange(cfg *_DNS_Config, c Conn, name string) (m *_DNS_Msg, err os.Error) {
	if len(name) >= 256 {
		return nil, &DNSError{"name too long", name, ""}
	}
	out := new(_DNS_Msg)
	out.id = 0x1234
	out.question = []_DNS_Question{
		_DNS_Question{name, _DNS_TypeA, _DNS_ClassINET},
	}
	out.recursion_desired = true
	msg, ok := out.Pack()
	if !ok {
		return nil, &DNSError{"internal error - cannot pack message", name, ""}
	}

	for attempt := 0; attempt < cfg.attempts; attempt++ {
		n, err := c.Write(msg)
		if err != nil {
			return nil, err
		}

		c.SetReadTimeout(1e9) // nanoseconds

		buf := make([]byte, 2000) // More than enough.
		n, err = c.Read(buf)
		if isEAGAIN(err) {
			err = nil
			continue
		}
		if err != nil {
			return nil, err
		}
		buf = buf[0:n]
		in := new(_DNS_Msg)
		if !in.Unpack(buf) || in.id != out.id {
			continue
		}
		return in, nil
	}
	var server string
	if a := c.RemoteAddr(); a != nil {
		server = a.String()
	}
	return nil, &DNSError{"no answer from server", name, server}
}


// Find answer for name in dns message.
// On return, if err == nil, addrs != nil.
func answer(name, server string, dns *_DNS_Msg) (addrs []string, err *DNSError) {
	addrs = make([]string, 0, len(dns.answer))

	if dns.rcode == _DNS_RcodeNameError && dns.recursion_available {
		return nil, &DNSError{noSuchHost, name, ""}
	}
	if dns.rcode != _DNS_RcodeSuccess {
		// None of the error codes make sense
		// for the query we sent.  If we didn't get
		// a name error and we didn't get success,
		// the server is behaving incorrectly.
		return nil, &DNSError{"server misbehaving", name, server}
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
			if h.Class == _DNS_ClassINET && h.Name == name {
				switch h.Rrtype {
				case _DNS_TypeA:
					n := len(addrs)
					a := rr.(*_DNS_RR_A).A
					addrs = addrs[0 : n+1]
					addrs[n] = IPv4(byte(a>>24), byte(a>>16), byte(a>>8), byte(a)).String()
				case _DNS_TypeCNAME:
					// redirect to cname
					name = rr.(*_DNS_RR_CNAME).Cname
					continue Cname
				}
			}
		}
		if len(addrs) == 0 {
			return nil, &DNSError{noSuchHost, name, server}
		}
		return addrs, nil
	}

	return nil, &DNSError{"too many redirects", name, server}
}

// Do a lookup for a single name, which must be rooted
// (otherwise answer will not find the answers).
func tryOneName(cfg *_DNS_Config, name string) (addrs []string, err os.Error) {
	if len(cfg.servers) == 0 {
		return nil, &DNSError{"no DNS servers", name, ""}
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
		msg, merr := _Exchange(cfg, c, name)
		c.Close()
		if merr != nil {
			err = merr
			continue
		}
		var dnserr *DNSError
		addrs, dnserr = answer(name, server, msg)
		if dnserr != nil {
			err = dnserr
		} else {
			err = nil // nil os.Error, not nil *DNSError
		}
		if dnserr == nil || dnserr.Error == noSuchHost {
			break
		}
	}
	return
}

var cfg *_DNS_Config
var dnserr os.Error

func loadConfig() { cfg, dnserr = _DNS_ReadConfig() }

func isDomainName(s string) bool {
	// Requirements on DNS name:
	//	* must not be empty.
	//	* must be alphanumeric plus - and .
	//	* each of the dot-separated elements must begin
	//	  and end with a letter or digit.
	//	  RFC 1035 required the element to begin with a letter,
	//	  but RFC 3696 says this has been relaxed to allow digits too.
	//	  still, there must be a letter somewhere in the entire name.
	if len(s) == 0 {
		return false
	}
	if s[len(s)-1] != '.' { // simplify checking loop: make name end in dot
		s += "."
	}

	last := byte('.')
	ok := false // ok once we've seen a letter
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		default:
			return false
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
			ok = true
		case '0' <= c && c <= '9':
			// fine
		case c == '-':
			// byte before dash cannot be dot
			if last == '.' {
				return false
			}
		case c == '.':
			// byte before dot cannot be dot, dash
			if last == '.' || last == '-' {
				return false
			}
		}
		last = c
	}

	return ok
}

// LookupHost looks up the host name using the local DNS resolver.
// It returns the canonical name for the host and an array of that
// host's addresses.
func LookupHost(name string) (cname string, addrs []string, err os.Error) {
	if !isDomainName(name) {
		return name, nil, &DNSError{"invalid domain name", name, ""}
	}
	once.Do(loadConfig)
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
		addrs, err = tryOneName(cfg, rname)
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
		addrs, err = tryOneName(cfg, rname)
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
	addrs, err = tryOneName(cfg, rname)
	if err == nil {
		cname = rname
		return
	}
	return
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"math/rand"
	"sort"
)

// DNSError represents a DNS lookup error.
type DNSError struct {
	Err       string // description of the error
	Name      string // name looked for
	Server    string // server used
	IsTimeout bool   // if true, timed out; not all timeouts set this
}

func (e *DNSError) Error() string {
	if e == nil {
		return "<nil>"
	}
	s := "lookup " + e.Name
	if e.Server != "" {
		s += " on " + e.Server
	}
	s += ": " + e.Err
	return s
}

// Timeout reports whether the DNS lookup is known to have timed out.
// This is not always known; a DNS lookup may fail due to a timeout
// and return a DNSError for which Timeout returns false.
func (e *DNSError) Timeout() bool { return e.IsTimeout }

// Temporary reports whether the DNS error is known to be temporary.
// This is not always known; a DNS lookup may fail due to a temporary
// error and return a DNSError for which Temporary returns false.
func (e *DNSError) Temporary() bool { return e.IsTimeout }

const noSuchHost = "no such host"

// reverseaddr returns the in-addr.arpa. or ip6.arpa. hostname of the IP
// address addr suitable for rDNS (PTR) record lookup or an error if it fails
// to parse the IP address.
func reverseaddr(addr string) (arpa string, err error) {
	ip := ParseIP(addr)
	if ip == nil {
		return "", &DNSError{Err: "unrecognized address", Name: addr}
	}
	if ip.To4() != nil {
		return uitoa(uint(ip[15])) + "." + uitoa(uint(ip[14])) + "." + uitoa(uint(ip[13])) + "." +
			uitoa(uint(ip[12])) + ".in-addr.arpa.", nil
	}
	// Must be IPv6
	buf := make([]byte, 0, len(ip)*4+len("ip6.arpa."))
	// Add it, in reverse, to the buffer
	for i := len(ip) - 1; i >= 0; i-- {
		v := ip[i]
		buf = append(buf, hexDigit[v&0xF])
		buf = append(buf, '.')
		buf = append(buf, hexDigit[v>>4])
		buf = append(buf, '.')
	}
	// Append "ip6.arpa." and return (buf already has the final .)
	buf = append(buf, "ip6.arpa."...)
	return string(buf), nil
}

// Find answer for name in dns message.
// On return, if err == nil, addrs != nil.
func answer(name, server string, dns *dnsMsg, qtype uint16) (cname string, addrs []dnsRR, err error) {
	addrs = make([]dnsRR, 0, len(dns.answer))

	if dns.rcode == dnsRcodeNameError && dns.recursion_available {
		return "", nil, &DNSError{Err: noSuchHost, Name: name}
	}
	if dns.rcode != dnsRcodeSuccess {
		// None of the error codes make sense
		// for the query we sent.  If we didn't get
		// a name error and we didn't get success,
		// the server is behaving incorrectly.
		return "", nil, &DNSError{Err: "server misbehaving", Name: name, Server: server}
	}

	// Look for the name.
	// Presotto says it's okay to assume that servers listed in
	// /etc/resolv.conf are recursive resolvers.
	// We asked for recursion, so it should have included
	// all the answers we need in this one packet.
Cname:
	for cnameloop := 0; cnameloop < 10; cnameloop++ {
		addrs = addrs[0:0]
		for _, rr := range dns.answer {
			if _, justHeader := rr.(*dnsRR_Header); justHeader {
				// Corrupt record: we only have a
				// header. That header might say it's
				// of type qtype, but we don't
				// actually have it. Skip.
				continue
			}
			h := rr.Header()
			if h.Class == dnsClassINET && equalASCIILabel(h.Name, name) {
				switch h.Rrtype {
				case qtype:
					addrs = append(addrs, rr)
				case dnsTypeCNAME:
					// redirect to cname
					name = rr.(*dnsRR_CNAME).Cname
					continue Cname
				}
			}
		}
		if len(addrs) == 0 {
			return "", nil, &DNSError{Err: noSuchHost, Name: name, Server: server}
		}
		return name, addrs, nil
	}

	return "", nil, &DNSError{Err: "too many redirects", Name: name, Server: server}
}

func equalASCIILabel(x, y string) bool {
	if len(x) != len(y) {
		return false
	}
	for i := 0; i < len(x); i++ {
		a := x[i]
		b := y[i]
		if 'A' <= a && a <= 'Z' {
			a += 0x20
		}
		if 'A' <= b && b <= 'Z' {
			b += 0x20
		}
		if a != b {
			return false
		}
	}
	return true
}

func isDomainName(s string) bool {
	// See RFC 1035, RFC 3696.
	if len(s) == 0 {
		return false
	}
	if len(s) > 255 {
		return false
	}

	last := byte('.')
	ok := false // Ok once we've seen a letter.
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
			// Byte before dash cannot be dot.
			if last == '.' {
				return false
			}
			partlen++
		case c == '.':
			// Byte before dot cannot be dot, dash.
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
	if last == '-' || partlen > 63 {
		return false
	}

	return ok
}

// An SRV represents a single DNS SRV record.
type SRV struct {
	Target   string
	Port     uint16
	Priority uint16
	Weight   uint16
}

// byPriorityWeight sorts SRV records by ascending priority and weight.
type byPriorityWeight []*SRV

func (s byPriorityWeight) Len() int { return len(s) }

func (s byPriorityWeight) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

func (s byPriorityWeight) Less(i, j int) bool {
	return s[i].Priority < s[j].Priority ||
		(s[i].Priority == s[j].Priority && s[i].Weight < s[j].Weight)
}

// shuffleByWeight shuffles SRV records by weight using the algorithm
// described in RFC 2782.
func (addrs byPriorityWeight) shuffleByWeight() {
	sum := 0
	for _, addr := range addrs {
		sum += int(addr.Weight)
	}
	for sum > 0 && len(addrs) > 1 {
		s := 0
		n := rand.Intn(sum)
		for i := range addrs {
			s += int(addrs[i].Weight)
			if s > n {
				if i > 0 {
					addrs[0], addrs[i] = addrs[i], addrs[0]
				}
				break
			}
		}
		sum -= int(addrs[0].Weight)
		addrs = addrs[1:]
	}
}

// sort reorders SRV records as specified in RFC 2782.
func (addrs byPriorityWeight) sort() {
	sort.Sort(addrs)
	i := 0
	for j := 1; j < len(addrs); j++ {
		if addrs[i].Priority != addrs[j].Priority {
			addrs[i:j].shuffleByWeight()
			i = j
		}
	}
	addrs[i:].shuffleByWeight()
}

// An MX represents a single DNS MX record.
type MX struct {
	Host string
	Pref uint16
}

// byPref implements sort.Interface to sort MX records by preference
type byPref []*MX

func (s byPref) Len() int { return len(s) }

func (s byPref) Less(i, j int) bool { return s[i].Pref < s[j].Pref }

func (s byPref) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

// sort reorders MX records as specified in RFC 5321.
func (s byPref) sort() {
	for i := range s {
		j := rand.Intn(i + 1)
		s[i], s[j] = s[j], s[i]
	}
	sort.Sort(s)
}

// An NS represents a single DNS NS record.
type NS struct {
	Host string
}

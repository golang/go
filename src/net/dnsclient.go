// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/bytealg"
	"internal/itoa"
	"sort"

	"golang.org/x/net/dns/dnsmessage"
)

// provided by runtime
func fastrand() uint32

func randInt() int {
	x, y := fastrand(), fastrand()    // 32-bit halves
	u := uint(x)<<31 ^ uint(int32(y)) // full uint, even on 64-bit systems; avoid 32-bit shift on 32-bit systems
	i := int(u >> 1)                  // clear sign bit, even on 32-bit systems
	return i
}

func randIntn(n int) int {
	return randInt() % n
}

// reverseaddr returns the in-addr.arpa. or ip6.arpa. hostname of the IP
// address addr suitable for rDNS (PTR) record lookup or an error if it fails
// to parse the IP address.
func reverseaddr(addr string) (arpa string, err error) {
	ip := ParseIP(addr)
	if ip == nil {
		return "", &DNSError{Err: "unrecognized address", Name: addr}
	}
	if ip.To4() != nil {
		return itoa.Uitoa(uint(ip[15])) + "." + itoa.Uitoa(uint(ip[14])) + "." + itoa.Uitoa(uint(ip[13])) + "." + itoa.Uitoa(uint(ip[12])) + ".in-addr.arpa.", nil
	}
	// Must be IPv6
	buf := make([]byte, 0, len(ip)*4+len("ip6.arpa."))
	// Add it, in reverse, to the buffer
	for i := len(ip) - 1; i >= 0; i-- {
		v := ip[i]
		buf = append(buf, hexDigit[v&0xF],
			'.',
			hexDigit[v>>4],
			'.')
	}
	// Append "ip6.arpa." and return (buf already has the final .)
	buf = append(buf, "ip6.arpa."...)
	return string(buf), nil
}

func equalASCIIName(x, y dnsmessage.Name) bool {
	if x.Length != y.Length {
		return false
	}
	for i := 0; i < int(x.Length); i++ {
		a := x.Data[i]
		b := y.Data[i]
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

// isDomainName checks if a string is a presentation-format domain name
// (currently restricted to hostname-compatible "preferred name" LDH labels and
// SRV-like "underscore labels"; see golang.org/issue/12421).
func isDomainName(s string) bool {
	// The root domain name is valid. See golang.org/issue/45715.
	if s == "." {
		return true
	}

	// See RFC 1035, RFC 3696.
	// Presentation format has dots before every label except the first, and the
	// terminal empty label is optional here because we assume fully-qualified
	// (absolute) input. We must therefore reserve space for the first and last
	// labels' length octets in wire format, where they are necessary and the
	// maximum total length is 255.
	// So our _effective_ maximum is 253, but 254 is not rejected if the last
	// character is a dot.
	l := len(s)
	if l == 0 || l > 254 || l == 254 && s[l-1] != '.' {
		return false
	}

	last := byte('.')
	nonNumeric := false // true once we've seen a letter or hyphen
	partlen := 0
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		default:
			return false
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z' || c == '_':
			nonNumeric = true
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
			nonNumeric = true
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

	return nonNumeric
}

// absDomainName returns an absolute domain name which ends with a
// trailing dot to match pure Go reverse resolver and all other lookup
// routines.
// See golang.org/issue/12189.
// But we don't want to add dots for local names from /etc/hosts.
// It's hard to tell so we settle on the heuristic that names without dots
// (like "localhost" or "myhost") do not get trailing dots, but any other
// names do.
func absDomainName(s string) string {
	if bytealg.IndexByteString(s, '.') != -1 && s[len(s)-1] != '.' {
		s += "."
	}
	return s
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
func (s byPriorityWeight) Less(i, j int) bool {
	return s[i].Priority < s[j].Priority || (s[i].Priority == s[j].Priority && s[i].Weight < s[j].Weight)
}
func (s byPriorityWeight) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

// shuffleByWeight shuffles SRV records by weight using the algorithm
// described in RFC 2782.
func (addrs byPriorityWeight) shuffleByWeight() {
	sum := 0
	for _, addr := range addrs {
		sum += int(addr.Weight)
	}
	for sum > 0 && len(addrs) > 1 {
		s := 0
		n := randIntn(sum)
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

func (s byPref) Len() int           { return len(s) }
func (s byPref) Less(i, j int) bool { return s[i].Pref < s[j].Pref }
func (s byPref) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// sort reorders MX records as specified in RFC 5321.
func (s byPref) sort() {
	for i := range s {
		j := randIntn(i + 1)
		s[i], s[j] = s[j], s[i]
	}
	sort.Sort(s)
}

// An NS represents a single DNS NS record.
type NS struct {
	Host string
}

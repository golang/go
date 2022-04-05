// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

// Minimal RFC 6724 address selection.

package net

import "sort"

func sortByRFC6724(addrs []IPAddr) {
	if len(addrs) < 2 {
		return
	}
	sortByRFC6724withSrcs(addrs, srcAddrs(addrs))
}

func sortByRFC6724withSrcs(addrs []IPAddr, srcs []IP) {
	if len(addrs) != len(srcs) {
		panic("internal error")
	}
	addrAttr := make([]ipAttr, len(addrs))
	srcAttr := make([]ipAttr, len(srcs))
	for i, v := range addrs {
		addrAttr[i] = ipAttrOf(v.IP)
		srcAttr[i] = ipAttrOf(srcs[i])
	}
	sort.Stable(&byRFC6724{
		addrs:    addrs,
		addrAttr: addrAttr,
		srcs:     srcs,
		srcAttr:  srcAttr,
	})
}

// srcsAddrs tries to UDP-connect to each address to see if it has a
// route. (This doesn't send any packets). The destination port
// number is irrelevant.
func srcAddrs(addrs []IPAddr) []IP {
	srcs := make([]IP, len(addrs))
	dst := UDPAddr{Port: 9}
	for i := range addrs {
		dst.IP = addrs[i].IP
		dst.Zone = addrs[i].Zone
		c, err := DialUDP("udp", nil, &dst)
		if err == nil {
			if src, ok := c.LocalAddr().(*UDPAddr); ok {
				srcs[i] = src.IP
			}
			c.Close()
		}
	}
	return srcs
}

type ipAttr struct {
	Scope      scope
	Precedence uint8
	Label      uint8
}

func ipAttrOf(ip IP) ipAttr {
	if ip == nil {
		return ipAttr{}
	}
	match := rfc6724policyTable.Classify(ip)
	return ipAttr{
		Scope:      classifyScope(ip),
		Precedence: match.Precedence,
		Label:      match.Label,
	}
}

type byRFC6724 struct {
	addrs    []IPAddr // addrs to sort
	addrAttr []ipAttr
	srcs     []IP // or nil if unreachable
	srcAttr  []ipAttr
}

func (s *byRFC6724) Len() int { return len(s.addrs) }

func (s *byRFC6724) Swap(i, j int) {
	s.addrs[i], s.addrs[j] = s.addrs[j], s.addrs[i]
	s.srcs[i], s.srcs[j] = s.srcs[j], s.srcs[i]
	s.addrAttr[i], s.addrAttr[j] = s.addrAttr[j], s.addrAttr[i]
	s.srcAttr[i], s.srcAttr[j] = s.srcAttr[j], s.srcAttr[i]
}

// Less reports whether i is a better destination address for this
// host than j.
//
// The algorithm and variable names comes from RFC 6724 section 6.
func (s *byRFC6724) Less(i, j int) bool {
	DA := s.addrs[i].IP
	DB := s.addrs[j].IP
	SourceDA := s.srcs[i]
	SourceDB := s.srcs[j]
	attrDA := &s.addrAttr[i]
	attrDB := &s.addrAttr[j]
	attrSourceDA := &s.srcAttr[i]
	attrSourceDB := &s.srcAttr[j]

	const preferDA = true
	const preferDB = false

	// Rule 1: Avoid unusable destinations.
	// If DB is known to be unreachable or if Source(DB) is undefined, then
	// prefer DA.  Similarly, if DA is known to be unreachable or if
	// Source(DA) is undefined, then prefer DB.
	if SourceDA == nil && SourceDB == nil {
		return false // "equal"
	}
	if SourceDB == nil {
		return preferDA
	}
	if SourceDA == nil {
		return preferDB
	}

	// Rule 2: Prefer matching scope.
	// If Scope(DA) = Scope(Source(DA)) and Scope(DB) <> Scope(Source(DB)),
	// then prefer DA.  Similarly, if Scope(DA) <> Scope(Source(DA)) and
	// Scope(DB) = Scope(Source(DB)), then prefer DB.
	if attrDA.Scope == attrSourceDA.Scope && attrDB.Scope != attrSourceDB.Scope {
		return preferDA
	}
	if attrDA.Scope != attrSourceDA.Scope && attrDB.Scope == attrSourceDB.Scope {
		return preferDB
	}

	// Rule 3: Avoid deprecated addresses.
	// If Source(DA) is deprecated and Source(DB) is not, then prefer DB.
	// Similarly, if Source(DA) is not deprecated and Source(DB) is
	// deprecated, then prefer DA.

	// TODO(bradfitz): implement? low priority for now.

	// Rule 4: Prefer home addresses.
	// If Source(DA) is simultaneously a home address and care-of address
	// and Source(DB) is not, then prefer DA.  Similarly, if Source(DB) is
	// simultaneously a home address and care-of address and Source(DA) is
	// not, then prefer DB.

	// TODO(bradfitz): implement? low priority for now.

	// Rule 5: Prefer matching label.
	// If Label(Source(DA)) = Label(DA) and Label(Source(DB)) <> Label(DB),
	// then prefer DA.  Similarly, if Label(Source(DA)) <> Label(DA) and
	// Label(Source(DB)) = Label(DB), then prefer DB.
	if attrSourceDA.Label == attrDA.Label &&
		attrSourceDB.Label != attrDB.Label {
		return preferDA
	}
	if attrSourceDA.Label != attrDA.Label &&
		attrSourceDB.Label == attrDB.Label {
		return preferDB
	}

	// Rule 6: Prefer higher precedence.
	// If Precedence(DA) > Precedence(DB), then prefer DA.  Similarly, if
	// Precedence(DA) < Precedence(DB), then prefer DB.
	if attrDA.Precedence > attrDB.Precedence {
		return preferDA
	}
	if attrDA.Precedence < attrDB.Precedence {
		return preferDB
	}

	// Rule 7: Prefer native transport.
	// If DA is reached via an encapsulating transition mechanism (e.g.,
	// IPv6 in IPv4) and DB is not, then prefer DB.  Similarly, if DB is
	// reached via encapsulation and DA is not, then prefer DA.

	// TODO(bradfitz): implement? low priority for now.

	// Rule 8: Prefer smaller scope.
	// If Scope(DA) < Scope(DB), then prefer DA.  Similarly, if Scope(DA) >
	// Scope(DB), then prefer DB.
	if attrDA.Scope < attrDB.Scope {
		return preferDA
	}
	if attrDA.Scope > attrDB.Scope {
		return preferDB
	}

	// Rule 9: Use longest matching prefix.
	// When DA and DB belong to the same address family (both are IPv6 or
	// both are IPv4 [but see below]): If CommonPrefixLen(Source(DA), DA) >
	// CommonPrefixLen(Source(DB), DB), then prefer DA.  Similarly, if
	// CommonPrefixLen(Source(DA), DA) < CommonPrefixLen(Source(DB), DB),
	// then prefer DB.
	//
	// However, applying this rule to IPv4 addresses causes
	// problems (see issues 13283 and 18518), so limit to IPv6.
	if DA.To4() == nil && DB.To4() == nil {
		commonA := commonPrefixLen(SourceDA, DA)
		commonB := commonPrefixLen(SourceDB, DB)

		if commonA > commonB {
			return preferDA
		}
		if commonA < commonB {
			return preferDB
		}
	}

	// Rule 10: Otherwise, leave the order unchanged.
	// If DA preceded DB in the original list, prefer DA.
	// Otherwise, prefer DB.
	return false // "equal"
}

type policyTableEntry struct {
	Prefix     *IPNet
	Precedence uint8
	Label      uint8
}

type policyTable []policyTableEntry

// RFC 6724 section 2.1.
var rfc6724policyTable = policyTable{
	{
		Prefix:     mustCIDR("::1/128"),
		Precedence: 50,
		Label:      0,
	},
	{
		Prefix:     mustCIDR("::/0"),
		Precedence: 40,
		Label:      1,
	},
	{
		// IPv4-compatible, etc.
		Prefix:     mustCIDR("::ffff:0:0/96"),
		Precedence: 35,
		Label:      4,
	},
	{
		// 6to4
		Prefix:     mustCIDR("2002::/16"),
		Precedence: 30,
		Label:      2,
	},
	{
		// Teredo
		Prefix:     mustCIDR("2001::/32"),
		Precedence: 5,
		Label:      5,
	},
	{
		Prefix:     mustCIDR("fc00::/7"),
		Precedence: 3,
		Label:      13,
	},
	{
		Prefix:     mustCIDR("::/96"),
		Precedence: 1,
		Label:      3,
	},
	{
		Prefix:     mustCIDR("fec0::/10"),
		Precedence: 1,
		Label:      11,
	},
	{
		Prefix:     mustCIDR("3ffe::/16"),
		Precedence: 1,
		Label:      12,
	},
}

func init() {
	sort.Sort(sort.Reverse(byMaskLength(rfc6724policyTable)))
}

// byMaskLength sorts policyTableEntry by the size of their Prefix.Mask.Size,
// from smallest mask, to largest.
type byMaskLength []policyTableEntry

func (s byMaskLength) Len() int      { return len(s) }
func (s byMaskLength) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s byMaskLength) Less(i, j int) bool {
	isize, _ := s[i].Prefix.Mask.Size()
	jsize, _ := s[j].Prefix.Mask.Size()
	return isize < jsize
}

// mustCIDR calls ParseCIDR and panics on any error, or if the network
// is not IPv6.
func mustCIDR(s string) *IPNet {
	ip, ipNet, err := ParseCIDR(s)
	if err != nil {
		panic(err.Error())
	}
	if len(ip) != IPv6len {
		panic("unexpected IP length")
	}
	return ipNet
}

// Classify returns the policyTableEntry of the entry with the longest
// matching prefix that contains ip.
// The table t must be sorted from largest mask size to smallest.
func (t policyTable) Classify(ip IP) policyTableEntry {
	for _, ent := range t {
		if ent.Prefix.Contains(ip) {
			return ent
		}
	}
	return policyTableEntry{}
}

// RFC 6724 section 3.1.
type scope uint8

const (
	scopeInterfaceLocal scope = 0x1
	scopeLinkLocal      scope = 0x2
	scopeAdminLocal     scope = 0x4
	scopeSiteLocal      scope = 0x5
	scopeOrgLocal       scope = 0x8
	scopeGlobal         scope = 0xe
)

func classifyScope(ip IP) scope {
	if ip.IsLoopback() || ip.IsLinkLocalUnicast() {
		return scopeLinkLocal
	}
	ipv6 := len(ip) == IPv6len && ip.To4() == nil
	if ipv6 && ip.IsMulticast() {
		return scope(ip[1] & 0xf)
	}
	// Site-local addresses are defined in RFC 3513 section 2.5.6
	// (and deprecated in RFC 3879).
	if ipv6 && ip[0] == 0xfe && ip[1]&0xc0 == 0xc0 {
		return scopeSiteLocal
	}
	return scopeGlobal
}

// commonPrefixLen reports the length of the longest prefix (looking
// at the most significant, or leftmost, bits) that the
// two addresses have in common, up to the length of a's prefix (i.e.,
// the portion of the address not including the interface ID).
//
// If a or b is an IPv4 address as an IPv6 address, the IPv4 addresses
// are compared (with max common prefix length of 32).
// If a and b are different IP versions, 0 is returned.
//
// See https://tools.ietf.org/html/rfc6724#section-2.2
func commonPrefixLen(a, b IP) (cpl int) {
	if a4 := a.To4(); a4 != nil {
		a = a4
	}
	if b4 := b.To4(); b4 != nil {
		b = b4
	}
	if len(a) != len(b) {
		return 0
	}
	// If IPv6, only up to the prefix (first 64 bits)
	if len(a) > 8 {
		a = a[:8]
		b = b[:8]
	}
	for len(a) > 0 {
		if a[0] == b[0] {
			cpl += 8
			a = a[1:]
			b = b[1:]
			continue
		}
		bits := 8
		ab, bb := a[0], b[0]
		for {
			ab >>= 1
			bb >>= 1
			bits--
			if ab == bb {
				cpl += bits
				return
			}
		}
	}
	return
}

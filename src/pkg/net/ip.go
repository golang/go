// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP address manipulations
//
// IPv4 addresses are 4 bytes; IPv6 addresses are 16 bytes.
// An IPv4 address can be converted to an IPv6 address by
// adding a canonical prefix (10 zeros, 2 0xFFs).
// This library accepts either size of byte slice but always
// returns 16-byte addresses.

package net

// IP address lengths (bytes).
const (
	IPv4len = 4
	IPv6len = 16
)

// An IP is a single IP address, a slice of bytes.
// Functions in this package accept either 4-byte (IPv4)
// or 16-byte (IPv6) slices as input.
//
// Note that in this documentation, referring to an
// IP address as an IPv4 address or an IPv6 address
// is a semantic property of the address, not just the
// length of the byte slice: a 16-byte slice can still
// be an IPv4 address.
type IP []byte

// An IP mask is an IP address.
type IPMask []byte

// An IPNet represents an IP network.
type IPNet struct {
	IP   IP     // network number
	Mask IPMask // network mask
}

// IPv4 returns the IP address (in 16-byte form) of the
// IPv4 address a.b.c.d.
func IPv4(a, b, c, d byte) IP {
	p := make(IP, IPv6len)
	copy(p, v4InV6Prefix)
	p[12] = a
	p[13] = b
	p[14] = c
	p[15] = d
	return p
}

var v4InV6Prefix = []byte{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff}

// IPv4Mask returns the IP mask (in 4-byte form) of the
// IPv4 mask a.b.c.d.
func IPv4Mask(a, b, c, d byte) IPMask {
	p := make(IPMask, IPv4len)
	p[0] = a
	p[1] = b
	p[2] = c
	p[3] = d
	return p
}

// CIDRMask returns an IPMask consisting of `ones' 1 bits
// followed by 0s up to a total length of `bits' bits.
// For a mask of this form, CIDRMask is the inverse of IPMask.Size.
func CIDRMask(ones, bits int) IPMask {
	if bits != 8*IPv4len && bits != 8*IPv6len {
		return nil
	}
	if ones < 0 || ones > bits {
		return nil
	}
	l := bits / 8
	m := make(IPMask, l)
	n := uint(ones)
	for i := 0; i < l; i++ {
		if n >= 8 {
			m[i] = 0xff
			n -= 8
			continue
		}
		m[i] = ^byte(0xff >> n)
		n = 0
	}
	return m
}

// Well-known IPv4 addresses
var (
	IPv4bcast     = IPv4(255, 255, 255, 255) // broadcast
	IPv4allsys    = IPv4(224, 0, 0, 1)       // all systems
	IPv4allrouter = IPv4(224, 0, 0, 2)       // all routers
	IPv4zero      = IPv4(0, 0, 0, 0)         // all zeros
)

// Well-known IPv6 addresses
var (
	IPv6zero                   = IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	IPv6unspecified            = IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
	IPv6loopback               = IP{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1}
	IPv6interfacelocalallnodes = IP{0xff, 0x01, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01}
	IPv6linklocalallnodes      = IP{0xff, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01}
	IPv6linklocalallrouters    = IP{0xff, 0x02, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02}
)

// IsUnspecified returns true if ip is an unspecified address.
func (ip IP) IsUnspecified() bool {
	if ip.Equal(IPv4zero) || ip.Equal(IPv6unspecified) {
		return true
	}
	return false
}

// IsLoopback returns true if ip is a loopback address.
func (ip IP) IsLoopback() bool {
	if ip4 := ip.To4(); ip4 != nil && ip4[0] == 127 {
		return true
	}
	return ip.Equal(IPv6loopback)
}

// IsMulticast returns true if ip is a multicast address.
func (ip IP) IsMulticast() bool {
	if ip4 := ip.To4(); ip4 != nil && ip4[0]&0xf0 == 0xe0 {
		return true
	}
	return ip[0] == 0xff
}

// IsInterfaceLinkLocalMulticast returns true if ip is
// an interface-local multicast address.
func (ip IP) IsInterfaceLocalMulticast() bool {
	return len(ip) == IPv6len && ip[0] == 0xff && ip[1]&0x0f == 0x01
}

// IsLinkLocalMulticast returns true if ip is a link-local
// multicast address.
func (ip IP) IsLinkLocalMulticast() bool {
	if ip4 := ip.To4(); ip4 != nil && ip4[0] == 224 && ip4[1] == 0 && ip4[2] == 0 {
		return true
	}
	return ip[0] == 0xff && ip[1]&0x0f == 0x02
}

// IsLinkLocalUnicast returns true if ip is a link-local
// unicast address.
func (ip IP) IsLinkLocalUnicast() bool {
	if ip4 := ip.To4(); ip4 != nil && ip4[0] == 169 && ip4[1] == 254 {
		return true
	}
	return ip[0] == 0xfe && ip[1]&0xc0 == 0x80
}

// IsGlobalUnicast returns true if ip is a global unicast
// address.
func (ip IP) IsGlobalUnicast() bool {
	return !ip.IsUnspecified() &&
		!ip.IsLoopback() &&
		!ip.IsMulticast() &&
		!ip.IsLinkLocalUnicast()
}

// Is p all zeros?
func isZeros(p IP) bool {
	for i := 0; i < len(p); i++ {
		if p[i] != 0 {
			return false
		}
	}
	return true
}

// To4 converts the IPv4 address ip to a 4-byte representation.
// If ip is not an IPv4 address, To4 returns nil.
func (ip IP) To4() IP {
	if len(ip) == IPv4len {
		return ip
	}
	if len(ip) == IPv6len &&
		isZeros(ip[0:10]) &&
		ip[10] == 0xff &&
		ip[11] == 0xff {
		return ip[12:16]
	}
	return nil
}

// To16 converts the IP address ip to a 16-byte representation.
// If ip is not an IP address (it is the wrong length), To16 returns nil.
func (ip IP) To16() IP {
	if len(ip) == IPv4len {
		return IPv4(ip[0], ip[1], ip[2], ip[3])
	}
	if len(ip) == IPv6len {
		return ip
	}
	return nil
}

// Default route masks for IPv4.
var (
	classAMask = IPv4Mask(0xff, 0, 0, 0)
	classBMask = IPv4Mask(0xff, 0xff, 0, 0)
	classCMask = IPv4Mask(0xff, 0xff, 0xff, 0)
)

// DefaultMask returns the default IP mask for the IP address ip.
// Only IPv4 addresses have default masks; DefaultMask returns
// nil if ip is not a valid IPv4 address.
func (ip IP) DefaultMask() IPMask {
	if ip = ip.To4(); ip == nil {
		return nil
	}
	switch true {
	case ip[0] < 0x80:
		return classAMask
	case ip[0] < 0xC0:
		return classBMask
	default:
		return classCMask
	}
}

func allFF(b []byte) bool {
	for _, c := range b {
		if c != 0xff {
			return false
		}
	}
	return true
}

// Mask returns the result of masking the IP address ip with mask.
func (ip IP) Mask(mask IPMask) IP {
	if len(mask) == IPv6len && len(ip) == IPv4len && allFF(mask[:12]) {
		mask = mask[12:]
	}
	if len(mask) == IPv4len && len(ip) == IPv6len && bytesEqual(ip[:12], v4InV6Prefix) {
		ip = ip[12:]
	}
	n := len(ip)
	if n != len(mask) {
		return nil
	}
	out := make(IP, n)
	for i := 0; i < n; i++ {
		out[i] = ip[i] & mask[i]
	}
	return out
}

// String returns the string form of the IP address ip.
// If the address is an IPv4 address, the string representation
// is dotted decimal ("74.125.19.99").  Otherwise the representation
// is IPv6 ("2001:4860:0:2001::68").
func (ip IP) String() string {
	p := ip

	if len(ip) == 0 {
		return "<nil>"
	}

	// If IPv4, use dotted notation.
	if p4 := p.To4(); len(p4) == IPv4len {
		return itod(uint(p4[0])) + "." +
			itod(uint(p4[1])) + "." +
			itod(uint(p4[2])) + "." +
			itod(uint(p4[3]))
	}
	if len(p) != IPv6len {
		return "?"
	}

	// Find longest run of zeros.
	e0 := -1
	e1 := -1
	for i := 0; i < IPv6len; i += 2 {
		j := i
		for j < IPv6len && p[j] == 0 && p[j+1] == 0 {
			j += 2
		}
		if j > i && j-i > e1-e0 {
			e0 = i
			e1 = j
		}
	}
	// The symbol "::" MUST NOT be used to shorten just one 16 bit 0 field.
	if e1-e0 <= 2 {
		e0 = -1
		e1 = -1
	}

	// Print with possible :: in place of run of zeros
	var s string
	for i := 0; i < IPv6len; i += 2 {
		if i == e0 {
			s += "::"
			i = e1
			if i >= IPv6len {
				break
			}
		} else if i > 0 {
			s += ":"
		}
		s += itox((uint(p[i])<<8)|uint(p[i+1]), 1)
	}
	return s
}

// Equal returns true if ip and x are the same IP address.
// An IPv4 address and that same address in IPv6 form are
// considered to be equal.
func (ip IP) Equal(x IP) bool {
	if len(ip) == len(x) {
		return bytesEqual(ip, x)
	}
	if len(ip) == IPv4len && len(x) == IPv6len {
		return bytesEqual(x[0:12], v4InV6Prefix) && bytesEqual(ip, x[12:])
	}
	if len(ip) == IPv6len && len(x) == IPv4len {
		return bytesEqual(ip[0:12], v4InV6Prefix) && bytesEqual(ip[12:], x)
	}
	return false
}

func bytesEqual(x, y []byte) bool {
	if len(x) != len(y) {
		return false
	}
	for i, b := range x {
		if y[i] != b {
			return false
		}
	}
	return true
}

// If mask is a sequence of 1 bits followed by 0 bits,
// return the number of 1 bits.
func simpleMaskLength(mask IPMask) int {
	var n int
	for i, v := range mask {
		if v == 0xff {
			n += 8
			continue
		}
		// found non-ff byte
		// count 1 bits
		for v&0x80 != 0 {
			n++
			v <<= 1
		}
		// rest must be 0 bits
		if v != 0 {
			return -1
		}
		for i++; i < len(mask); i++ {
			if mask[i] != 0 {
				return -1
			}
		}
		break
	}
	return n
}

// Size returns the number of leading ones and total bits in the mask.
// If the mask is not in the canonical form--ones followed by zeros--then
// Size returns 0, 0.
func (m IPMask) Size() (ones, bits int) {
	ones, bits = simpleMaskLength(m), len(m)*8
	if ones == -1 {
		return 0, 0
	}
	return
}

// String returns the hexadecimal form of m, with no punctuation.
func (m IPMask) String() string {
	s := ""
	for _, b := range m {
		s += itox(uint(b), 2)
	}
	if len(s) == 0 {
		return "<nil>"
	}
	return s
}

func networkNumberAndMask(n *IPNet) (ip IP, m IPMask) {
	if ip = n.IP.To4(); ip == nil {
		ip = n.IP
		if len(ip) != IPv6len {
			return nil, nil
		}
	}
	m = n.Mask
	switch len(m) {
	case IPv4len:
		if len(ip) != IPv4len {
			return nil, nil
		}
	case IPv6len:
		if len(ip) == IPv4len {
			m = m[12:]
		}
	default:
		return nil, nil
	}
	return
}

// Contains reports whether the network includes ip.
func (n *IPNet) Contains(ip IP) bool {
	nn, m := networkNumberAndMask(n)
	if x := ip.To4(); x != nil {
		ip = x
	}
	l := len(ip)
	if l != len(nn) {
		return false
	}
	for i := 0; i < l; i++ {
		if nn[i]&m[i] != ip[i]&m[i] {
			return false
		}
	}
	return true
}

// Network returns the address's network name, "ip+net".
func (n *IPNet) Network() string { return "ip+net" }

// String returns the CIDR notation of n like "192.168.100.1/24"
// or "2001:DB8::/48" as defined in RFC 4632 and RFC 4291.
// If the mask is not in the canonical form, it returns the
// string which consists of an IP address, followed by a slash
// character and a mask expressed as hexadecimal form with no
// punctuation like "192.168.100.1/c000ff00".
func (n *IPNet) String() string {
	nn, m := networkNumberAndMask(n)
	if nn == nil || m == nil {
		return "<nil>"
	}
	l := simpleMaskLength(m)
	if l == -1 {
		return nn.String() + "/" + m.String()
	}
	return nn.String() + "/" + itod(uint(l))
}

// Parse IPv4 address (d.d.d.d).
func parseIPv4(s string) IP {
	var p [IPv4len]byte
	i := 0
	for j := 0; j < IPv4len; j++ {
		if i >= len(s) {
			// Missing octets.
			return nil
		}
		if j > 0 {
			if s[i] != '.' {
				return nil
			}
			i++
		}
		var (
			n  int
			ok bool
		)
		n, i, ok = dtoi(s, i)
		if !ok || n > 0xFF {
			return nil
		}
		p[j] = byte(n)
	}
	if i != len(s) {
		return nil
	}
	return IPv4(p[0], p[1], p[2], p[3])
}

// parseIPv6 parses s as a literal IPv6 address described in RFC 4291
// and RFC 5952.  It can also parse a literal scoped IPv6 address with
// zone identifier which is described in RFC 4007 when zoneAllowed is
// true.
func parseIPv6(s string, zoneAllowed bool) (ip IP, zone string) {
	ip = make(IP, IPv6len)
	ellipsis := -1 // position of ellipsis in p
	i := 0         // index in string s

	if zoneAllowed {
		s, zone = splitHostZone(s)
	}

	// Might have leading ellipsis
	if len(s) >= 2 && s[0] == ':' && s[1] == ':' {
		ellipsis = 0
		i = 2
		// Might be only ellipsis
		if i == len(s) {
			return ip, zone
		}
	}

	// Loop, parsing hex numbers followed by colon.
	j := 0
	for j < IPv6len {
		// Hex number.
		n, i1, ok := xtoi(s, i)
		if !ok || n > 0xFFFF {
			return nil, zone
		}

		// If followed by dot, might be in trailing IPv4.
		if i1 < len(s) && s[i1] == '.' {
			if ellipsis < 0 && j != IPv6len-IPv4len {
				// Not the right place.
				return nil, zone
			}
			if j+IPv4len > IPv6len {
				// Not enough room.
				return nil, zone
			}
			ip4 := parseIPv4(s[i:])
			if ip4 == nil {
				return nil, zone
			}
			ip[j] = ip4[12]
			ip[j+1] = ip4[13]
			ip[j+2] = ip4[14]
			ip[j+3] = ip4[15]
			i = len(s)
			j += IPv4len
			break
		}

		// Save this 16-bit chunk.
		ip[j] = byte(n >> 8)
		ip[j+1] = byte(n)
		j += 2

		// Stop at end of string.
		i = i1
		if i == len(s) {
			break
		}

		// Otherwise must be followed by colon and more.
		if s[i] != ':' || i+1 == len(s) {
			return nil, zone
		}
		i++

		// Look for ellipsis.
		if s[i] == ':' {
			if ellipsis >= 0 { // already have one
				return nil, zone
			}
			ellipsis = j
			if i++; i == len(s) { // can be at end
				break
			}
		}
	}

	// Must have used entire string.
	if i != len(s) {
		return nil, zone
	}

	// If didn't parse enough, expand ellipsis.
	if j < IPv6len {
		if ellipsis < 0 {
			return nil, zone
		}
		n := IPv6len - j
		for k := j - 1; k >= ellipsis; k-- {
			ip[k+n] = ip[k]
		}
		for k := ellipsis + n - 1; k >= ellipsis; k-- {
			ip[k] = 0
		}
	}
	return ip, zone
}

// A ParseError represents a malformed text string and the type of string that was expected.
type ParseError struct {
	Type string
	Text string
}

func (e *ParseError) Error() string {
	return "invalid " + e.Type + ": " + e.Text
}

// ParseIP parses s as an IP address, returning the result.
// The string s can be in dotted decimal ("74.125.19.99")
// or IPv6 ("2001:4860:0:2001::68") form.
// If s is not a valid textual representation of an IP address,
// ParseIP returns nil.
func ParseIP(s string) IP {
	if ip := parseIPv4(s); ip != nil {
		return ip
	}
	ip, _ := parseIPv6(s, false)
	return ip
}

// ParseCIDR parses s as a CIDR notation IP address and mask,
// like "192.168.100.1/24" or "2001:DB8::/48", as defined in
// RFC 4632 and RFC 4291.
//
// It returns the IP address and the network implied by the IP
// and mask.  For example, ParseCIDR("192.168.100.1/16") returns
// the IP address 192.168.100.1 and the network 192.168.0.0/16.
func ParseCIDR(s string) (IP, *IPNet, error) {
	i := byteIndex(s, '/')
	if i < 0 {
		return nil, nil, &ParseError{"CIDR address", s}
	}
	addr, mask := s[:i], s[i+1:]
	iplen := IPv4len
	ip := parseIPv4(addr)
	if ip == nil {
		iplen = IPv6len
		ip, _ = parseIPv6(addr, false)
	}
	n, i, ok := dtoi(mask, 0)
	if ip == nil || !ok || i != len(mask) || n < 0 || n > 8*iplen {
		return nil, nil, &ParseError{"CIDR address", s}
	}
	m := CIDRMask(n, 8*iplen)
	return ip, &IPNet{IP: ip.Mask(m), Mask: m}, nil
}

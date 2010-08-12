// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// IP address manipulations
//
// IPv4 addresses are 4 bytes; IPv6 addresses are 16 bytes.
// An IPv4 address can be converted to an IPv6 address by
// adding a canonical prefix (10 zeros, 2 0xFFs).
// This library accepts either size of byte array but always
// returns 16-byte addresses.

package net

// IP address lengths (bytes).
const (
	IPv4len = 4
	IPv6len = 16
)

// An IP is a single IP address, an array of bytes.
// Functions in this package accept either 4-byte (IP v4)
// or 16-byte (IP v6) arrays as input.  Unless otherwise
// specified, functions in this package always return
// IP addresses in 16-byte form using the canonical
// embedding.
//
// Note that in this documentation, referring to an
// IP address as an IPv4 address or an IPv6 address
// is a semantic property of the address, not just the
// length of the byte array: a 16-byte array can still
// be an IPv4 address.
type IP []byte

// An IP mask is an IP address.
type IPMask []byte

// IPv4 returns the IP address (in 16-byte form) of the
// IPv4 address a.b.c.d.
func IPv4(a, b, c, d byte) IP {
	p := make(IP, IPv6len)
	for i := 0; i < 10; i++ {
		p[i] = 0
	}
	p[10] = 0xff
	p[11] = 0xff
	p[12] = a
	p[13] = b
	p[14] = c
	p[15] = d
	return p
}

// IPv4Mask returns the IP mask (in 16-byte form) of the
// IPv4 mask a.b.c.d.
func IPv4Mask(a, b, c, d byte) IPMask {
	p := make(IPMask, IPv6len)
	for i := 0; i < 12; i++ {
		p[i] = 0xff
	}
	p[12] = a
	p[13] = b
	p[14] = c
	p[15] = d
	return p
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
	IPzero = make(IP, IPv6len) // all zeros
)

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
	return nil // not reached
}

// Mask returns the result of masking the IP address ip with mask.
func (ip IP) Mask(mask IPMask) IP {
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

// Convert i to decimal string.
func itod(i uint) string {
	if i == 0 {
		return "0"
	}

	// Assemble decimal in reverse order.
	var b [32]byte
	bp := len(b)
	for ; i > 0; i /= 10 {
		bp--
		b[bp] = byte(i%10) + '0'
	}

	return string(b[bp:])
}

// Convert i to hexadecimal string.
func itox(i uint) string {
	if i == 0 {
		return "0"
	}

	// Assemble hexadecimal in reverse order.
	var b [32]byte
	bp := len(b)
	for ; i > 0; i /= 16 {
		bp--
		b[bp] = "0123456789abcdef"[byte(i%16)]
	}

	return string(b[bp:])
}

// String returns the string form of the IP address ip.
// If the address is an IPv4 address, the string representation
// is dotted decimal ("74.125.19.99").  Otherwise the representation
// is IPv6 ("2001:4860:0:2001::68").
func (ip IP) String() string {
	p := ip

	if len(ip) == 0 {
		return ""
	}

	// If IPv4, use dotted notation.
	if p4 := p.To4(); len(p4) == 4 {
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
	for i := 0; i < 16; i += 2 {
		j := i
		for j < 16 && p[j] == 0 && p[j+1] == 0 {
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
	for i := 0; i < 16; i += 2 {
		if i == e0 {
			s += "::"
			i = e1
			if i >= 16 {
				break
			}
		} else if i > 0 {
			s += ":"
		}
		s += itox((uint(p[i]) << 8) | uint(p[i+1]))
	}
	return s
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

// String returns the string representation of mask.
// If the mask is in the canonical form--ones followed by zeros--the
// string representation is just the decimal number of ones.
// If the mask is in a non-canonical form, it is formatted
// as an IP address.
func (mask IPMask) String() string {
	switch len(mask) {
	case 4:
		n := simpleMaskLength(mask)
		if n >= 0 {
			return itod(uint(n + (IPv6len-IPv4len)*8))
		}
	case 16:
		n := simpleMaskLength(mask)
		if n >= 12*8 {
			return itod(uint(n - 12*8))
		}
	}
	return IP(mask).String()
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

// Parse IPv6 address.  Many forms.
// The basic form is a sequence of eight colon-separated
// 16-bit hex numbers separated by colons,
// as in 0123:4567:89ab:cdef:0123:4567:89ab:cdef.
// Two exceptions:
//	* A run of zeros can be replaced with "::".
//	* The last 32 bits can be in IPv4 form.
// Thus, ::ffff:1.2.3.4 is the IPv4 address 1.2.3.4.
func parseIPv6(s string) IP {
	p := make(IP, 16)
	ellipsis := -1 // position of ellipsis in p
	i := 0         // index in string s

	// Might have leading ellipsis
	if len(s) >= 2 && s[0] == ':' && s[1] == ':' {
		ellipsis = 0
		i = 2
		// Might be only ellipsis
		if i == len(s) {
			return p
		}
	}

	// Loop, parsing hex numbers followed by colon.
	j := 0
L:
	for j < IPv6len {
		// Hex number.
		n, i1, ok := xtoi(s, i)
		if !ok || n > 0xFFFF {
			return nil
		}

		// If followed by dot, might be in trailing IPv4.
		if i1 < len(s) && s[i1] == '.' {
			if ellipsis < 0 && j != IPv6len-IPv4len {
				// Not the right place.
				return nil
			}
			if j+IPv4len > IPv6len {
				// Not enough room.
				return nil
			}
			p4 := parseIPv4(s[i:])
			if p4 == nil {
				return nil
			}
			p[j] = p4[12]
			p[j+1] = p4[13]
			p[j+2] = p4[14]
			p[j+3] = p4[15]
			i = len(s)
			j += 4
			break
		}

		// Save this 16-bit chunk.
		p[j] = byte(n >> 8)
		p[j+1] = byte(n)
		j += 2

		// Stop at end of string.
		i = i1
		if i == len(s) {
			break
		}

		// Otherwise must be followed by colon and more.
		if s[i] != ':' && i+1 == len(s) {
			return nil
		}
		i++

		// Look for ellipsis.
		if s[i] == ':' {
			if ellipsis >= 0 { // already have one
				return nil
			}
			ellipsis = j
			if i++; i == len(s) { // can be at end
				break
			}
		}
	}

	// Must have used entire string.
	if i != len(s) {
		return nil
	}

	// If didn't parse enough, expand ellipsis.
	if j < IPv6len {
		if ellipsis < 0 {
			return nil
		}
		n := IPv6len - j
		for k := j - 1; k >= ellipsis; k-- {
			p[k+n] = p[k]
		}
		for k := ellipsis + n - 1; k >= ellipsis; k-- {
			p[k] = 0
		}
	}
	return p
}

// ParseIP parses s as an IP address, returning the result.
// The string s can be in dotted decimal ("74.125.19.99")
// or IPv6 ("2001:4860:0:2001::68") form.
// If s is not a valid textual representation of an IP address,
// ParseIP returns nil.
func ParseIP(s string) IP {
	p := parseIPv4(s)
	if p != nil {
		return p
	}
	return parseIPv6(s)
}

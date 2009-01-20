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

import (
	"net"
)

const (
	IPv4len = 4;
	IPv6len = 16
)

// Make the 4 bytes into an IPv4 address (in IPv6 form)
func _MakeIPv4(a, b, c, d byte) []byte {
	p := make([]byte, IPv6len);
	for i := 0; i < 10; i++ {
		p[i] = 0
	}
	p[10] = 0xff;
	p[11] = 0xff;
	p[12] = a;
	p[13] = b;
	p[14] = c;
	p[15] = d;
	return p
}

// Well-known IP addresses
var IPv4bcast, IPv4allsys, IPv4allrouter, IPv4prefix, IPallbits, IPnoaddr []byte

func init() {
	IPv4bcast = _MakeIPv4(0xff, 0xff, 0xff, 0xff);
	IPv4allsys = _MakeIPv4(0xe0, 0x00, 0x00, 0x01);
	IPv4allrouter = _MakeIPv4(0xe0, 0x00, 0x00, 0x02);
	IPv4prefix = _MakeIPv4(0, 0, 0, 0);
	IPallbits = make([]byte, IPv6len);
	for i := 0; i < IPv6len; i++ {
		IPallbits[i] = 0xff
	}
	IPnoaddr = make([]byte, IPv6len);	// zeroed
}

// Is p all zeros?
func _IsZeros(p []byte) bool {
	for i := 0; i < len(p); i++ {
		if p[i] != 0 {
			return false
		}
	}
	return true
}

// Is p an IPv4 address (perhaps in IPv6 form)?
// If so, return the 4-byte V4 array.
func ToIPv4(p []byte) []byte {
	if len(p) == IPv4len {
		return p
	}
	if len(p) == IPv6len
	&& _IsZeros(p[0:10])
	&& p[10] == 0xff
	&& p[11] == 0xff {
		return p[12:16]
	}
	return nil
}

// Convert p to IPv6 form.
func ToIPv6(p []byte) []byte {
	if len(p) == IPv4len {
		return _MakeIPv4(p[0], p[1], p[2], p[3])
	}
	if len(p) == IPv6len {
		return p
	}
	return nil
}

// Default route masks for IPv4.
var (
	ClassAMask = _MakeIPv4(0xff, 0, 0, 0);
	ClassBMask = _MakeIPv4(0xff, 0xff, 0, 0);
	ClassCMask = _MakeIPv4(0xff, 0xff, 0xff, 0);
)

func DefaultMask(p []byte) []byte {
	if p = ToIPv4(p); p == nil {
		return nil
	}
	switch true {
	case p[0] < 0x80:
		return ClassAMask;
	case p[0] < 0xC0:
		return ClassBMask;
	default:
		return ClassCMask;
	}
	return nil;	// not reached
}

// Apply mask to ip, returning new address.
func Mask(ip []byte, mask []byte) []byte {
	n := len(ip);
	if n != len(mask) {
		return nil
	}
	out := make([]byte, n);
	for i := 0; i < n; i++ {
		out[i] = ip[i] & mask[i];
	}
	return out
}

// Convert i to decimal string.
func itod(i uint) string {
	if i == 0 {
		return "0"
	}

	// Assemble decimal in reverse order.
	var b [32]byte;
	bp := len(b);
	for ; i > 0; i /= 10 {
		bp--;
		b[bp] = byte(i%10) + '0'
	}

	return string(b[bp:len(b)])
//	return string((&b)[bp:len(b)])
}

// Convert i to hexadecimal string.
func itox(i uint) string {
	if i == 0 {
		return "0"
	}

	// Assemble hexadecimal in reverse order.
	var b [32]byte;
	bp := len(b);
	for ; i > 0; i /= 16 {
		bp--;
		b[bp] = "0123456789abcdef"[byte(i%16)]
	}

	return string(b[bp:len(b)])
	// return string((&b)[bp:len(b)])
}

// Convert IP address to string.
func IPToString(p []byte) string {
	// If IPv4, use dotted notation.
	if p4 := ToIPv4(p); len(p4) == 4 {
		return itod(uint(p4[0]))+"."
			+itod(uint(p4[1]))+"."
			+itod(uint(p4[2]))+"."
			+itod(uint(p4[3]))
	}
	if len(p) != IPv6len {
		return "?"
	}

	// Find longest run of zeros.
	e0 := -1;
	e1 := -1;
	for i := 0; i < 16; i+=2 {
		j := i;
		for j < 16 && p[j] == 0 && p[j+1] == 0 {
			j += 2
		}
		if j > i && j - i > e1 - e0 {
			e0 = i;
			e1 = j
		}
	}

	// Print with possible :: in place of run of zeros
	var s string;
	for i := 0; i < 16; i += 2 {
		if i == e0 {
			s += "::";
			i = e1;
			if i >= 16 {
				break
			}
		} else if i > 0 {
			s += ":"
		}
		s += itox((uint(p[i])<<8) | uint(p[i+1]))
	}
	return s
}

// If mask is a sequence of 1 bits followed by 0 bits,
// return the number of 1 bits.
func _SimpleMaskLength(mask []byte) int {
	var i int;
	for i = 0; i < len(mask); i++ {
		if mask[i] != 0xFF {
			break
		}
	}
	n := 8*i;
	v := mask[i];
	for v & 0x80 != 0 {
		n++;
		v <<= 1
	}
	if v != 0 {
		return -1
	}
	for i++; i < len(mask); i++ {
		if mask[i] != 0 {
			return -1
		}
	}
	return n
}

func MaskToString(mask []byte) string {
	switch len(mask) {
	case 4:
		n := _SimpleMaskLength(mask);
		if n >= 0 {
			return itod(uint(n+(IPv6len-IPv4len)*8))
		}
	case 16:
		n := _SimpleMaskLength(mask);
		if n >= 0 {
			return itod(uint(n))
		}
	}
	return IPToString(mask)
}

// Parse IPv4 address (d.d.d.d).
func _ParseIPv4(s string) []byte {
	var p [IPv4len]byte;
	i := 0;
	for j := 0; j < IPv4len; j++ {
		if j > 0 {
			if s[i] != '.' {
				return nil
			}
			i++;
		}
		var (
			n int;
			ok bool
		)
		n, i, ok = _Dtoi(s, i);
		if !ok || n > 0xFF {
			return nil
		}
		p[j] = byte(n)
	}
	if i != len(s) {
		return nil
	}
	return _MakeIPv4(p[0], p[1], p[2], p[3])
}

// Parse IPv6 address.  Many forms.
// The basic form is a sequence of eight colon-separated
// 16-bit hex numbers separated by colons,
// as in 0123:4567:89ab:cdef:0123:4567:89ab:cdef.
// Two exceptions:
//	* A run of zeros can be replaced with "::".
//	* The last 32 bits can be in IPv4 form.
// Thus, ::ffff:1.2.3.4 is the IPv4 address 1.2.3.4.
func _ParseIPv6(s string) []byte {
	p := make([]byte, 16);
	ellipsis := -1;	// position of ellipsis in p
	i := 0;	// index in string s

	// Might have leading ellipsis
	if len(s) >= 2 && s[0] == ':' && s[1] == ':' {
		ellipsis = 0;
		i = 2;
		// Might be only ellipsis
		if i == len(s) {
			return p
		}
	}

	// Loop, parsing hex numbers followed by colon.
	j := 0;
L:	for j < IPv6len {
		// Hex number.
		n, i1, ok := _Xtoi(s, i);
		if !ok || n > 0xFFFF {
			return nil
		}

		// If followed by dot, might be in trailing IPv4.
		if i1 < len(s) && s[i1] == '.' {
			if ellipsis < 0 && j != IPv6len - IPv4len {
				// Not the right place.
				return nil
			}
			if j+IPv4len > IPv6len {
				// Not enough room.
				return nil
			}
			p4 := _ParseIPv4(s[i:len(s)]);
			if p4 == nil {
				return nil
			}
			// BUG: p[j:j+4] = p4
			p[j] = p4[12];
			p[j+1] = p4[13];
			p[j+2] = p4[14];
			p[j+3] = p4[15];
			i = len(s);
			j += 4;
			break
		}

		// Save this 16-bit chunk.
		p[j] = byte(n>>8);
		p[j+1] = byte(n);
		j += 2;

		// Stop at end of string.
		i = i1;
		if i == len(s) {
			break
		}

		// Otherwise must be followed by colon and more.
		if s[i] != ':' && i+1 == len(s) {
			return nil
		}
		i++;

		// Look for ellipsis.
		if s[i] == ':' {
			if ellipsis >= 0 {	// already have one
				return nil
			}
			ellipsis = j;
			if i++; i == len(s) {	// can be at end
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
		n := IPv6len - j;
		for k := j-1; k >= ellipsis; k-- {
			p[k+n] = p[k]
		}
		for k := ellipsis+n-1; k>=ellipsis; k-- {
			p[k] = 0
		}
	}
	return p
}

func ParseIP(s string) []byte {
	p := _ParseIPv4(s);
	if p != nil {
		return p
	}
	return _ParseIPv6(s)
}


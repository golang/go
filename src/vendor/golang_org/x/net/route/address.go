// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package route

import "runtime"

// An Addr represents an address associated with packet routing.
type Addr interface {
	// Family returns an address family.
	Family() int
}

// A LinkAddr represents a link-layer address.
type LinkAddr struct {
	Index int    // interface index when attached
	Name  string // interface name when attached
	Addr  []byte // link-layer address when attached
}

// Family implements the Family method of Addr interface.
func (a *LinkAddr) Family() int { return sysAF_LINK }

func parseLinkAddr(b []byte) (Addr, error) {
	if len(b) < 8 {
		return nil, errInvalidAddr
	}
	_, a, err := parseKernelLinkAddr(sysAF_LINK, b[4:])
	if err != nil {
		return nil, err
	}
	a.(*LinkAddr).Index = int(nativeEndian.Uint16(b[2:4]))
	return a, nil
}

// parseKernelLinkAddr parses b as a link-layer address in
// conventional BSD kernel form.
func parseKernelLinkAddr(_ int, b []byte) (int, Addr, error) {
	// The encoding looks like the following:
	// +----------------------------+
	// | Type             (1 octet) |
	// +----------------------------+
	// | Name length      (1 octet) |
	// +----------------------------+
	// | Address length   (1 octet) |
	// +----------------------------+
	// | Selector length  (1 octet) |
	// +----------------------------+
	// | Data            (variable) |
	// +----------------------------+
	//
	// On some platforms, all-bit-one of length field means "don't
	// care".
	nlen, alen, slen := int(b[1]), int(b[2]), int(b[3])
	if nlen == 0xff {
		nlen = 0
	}
	if alen == 0xff {
		alen = 0
	}
	if slen == 0xff {
		slen = 0
	}
	l := 4 + nlen + alen + slen
	if len(b) < l {
		return 0, nil, errInvalidAddr
	}
	data := b[4:]
	var name string
	var addr []byte
	if nlen > 0 {
		name = string(data[:nlen])
		data = data[nlen:]
	}
	if alen > 0 {
		addr = data[:alen]
		data = data[alen:]
	}
	return l, &LinkAddr{Name: name, Addr: addr}, nil
}

// An Inet4Addr represents an internet address for IPv4.
type Inet4Addr struct {
	IP [4]byte // IP address
}

// Family implements the Family method of Addr interface.
func (a *Inet4Addr) Family() int { return sysAF_INET }

// An Inet6Addr represents an internet address for IPv6.
type Inet6Addr struct {
	IP     [16]byte // IP address
	ZoneID int      // zone identifier
}

// Family implements the Family method of Addr interface.
func (a *Inet6Addr) Family() int { return sysAF_INET6 }

// parseInetAddr parses b as an internet address for IPv4 or IPv6.
func parseInetAddr(af int, b []byte) (Addr, error) {
	switch af {
	case sysAF_INET:
		if len(b) < 16 {
			return nil, errInvalidAddr
		}
		a := &Inet4Addr{}
		copy(a.IP[:], b[4:8])
		return a, nil
	case sysAF_INET6:
		if len(b) < 28 {
			return nil, errInvalidAddr
		}
		a := &Inet6Addr{ZoneID: int(nativeEndian.Uint32(b[24:28]))}
		copy(a.IP[:], b[8:24])
		if a.IP[0] == 0xfe && a.IP[1]&0xc0 == 0x80 || a.IP[0] == 0xff && (a.IP[1]&0x0f == 0x01 || a.IP[1]&0x0f == 0x02) {
			// KAME based IPv6 protocol stack usually
			// embeds the interface index in the
			// interface-local or link-local address as
			// the kernel-internal form.
			id := int(bigEndian.Uint16(a.IP[2:4]))
			if id != 0 {
				a.ZoneID = id
				a.IP[2], a.IP[3] = 0, 0
			}
		}
		return a, nil
	default:
		return nil, errInvalidAddr
	}
}

// parseKernelInetAddr parses b as an internet address in conventional
// BSD kernel form.
func parseKernelInetAddr(af int, b []byte) (int, Addr, error) {
	// The encoding looks similar to the NLRI encoding.
	// +----------------------------+
	// | Length           (1 octet) |
	// +----------------------------+
	// | Address prefix  (variable) |
	// +----------------------------+
	//
	// The differences between the kernel form and the NLRI
	// encoding are:
	//
	// - The length field of the kernel form indicates the prefix
	//   length in bytes, not in bits
	//
	// - In the kernel form, zero value of the length field
	//   doesn't mean 0.0.0.0/0 or ::/0
	//
	// - The kernel form appends leading bytes to the prefix field
	//   to make the <length, prefix> tuple to be conformed with
	//   the routing message boundary
	l := int(b[0])
	if runtime.GOOS == "darwin" {
		// On Darwn, an address in the kernel form is also
		// used as a message filler.
		if l == 0 || len(b) > roundup(l) {
			l = roundup(l)
		}
	} else {
		l = roundup(l)
	}
	if len(b) < l {
		return 0, nil, errInvalidAddr
	}
	// Don't reorder case expressions.
	// The case expressions for IPv6 must come first.
	const (
		off4 = 4 // offset of in_addr
		off6 = 8 // offset of in6_addr
	)
	switch {
	case b[0] == 28: // size of sockaddr_in6
		a := &Inet6Addr{}
		copy(a.IP[:], b[off6:off6+16])
		return int(b[0]), a, nil
	case af == sysAF_INET6:
		a := &Inet6Addr{}
		if l-1 < off6 {
			copy(a.IP[:], b[1:l])
		} else {
			copy(a.IP[:], b[l-off6:l])
		}
		return int(b[0]), a, nil
	case b[0] == 16: // size of sockaddr_in
		a := &Inet4Addr{}
		copy(a.IP[:], b[off4:off4+4])
		return int(b[0]), a, nil
	default: // an old fashion, AF_UNSPEC or unknown means AF_INET
		a := &Inet4Addr{}
		if l-1 < off4 {
			copy(a.IP[:], b[1:l])
		} else {
			copy(a.IP[:], b[l-off4:l])
		}
		return int(b[0]), a, nil
	}
}

// A DefaultAddr represents an address of various operating
// system-specific features.
type DefaultAddr struct {
	af  int
	Raw []byte // raw format of address
}

// Family implements the Family method of Addr interface.
func (a *DefaultAddr) Family() int { return a.af }

func parseDefaultAddr(b []byte) (Addr, error) {
	if len(b) < 2 || len(b) < int(b[0]) {
		return nil, errInvalidAddr
	}
	a := &DefaultAddr{af: int(b[1]), Raw: b[:b[0]]}
	return a, nil
}

func parseAddrs(attrs uint, fn func(int, []byte) (int, Addr, error), b []byte) ([]Addr, error) {
	var as [sysRTAX_MAX]Addr
	af := int(sysAF_UNSPEC)
	for i := uint(0); i < sysRTAX_MAX && len(b) >= roundup(0); i++ {
		if attrs&(1<<i) == 0 {
			continue
		}
		if i <= sysRTAX_BRD {
			switch b[1] {
			case sysAF_LINK:
				a, err := parseLinkAddr(b)
				if err != nil {
					return nil, err
				}
				as[i] = a
				l := roundup(int(b[0]))
				if len(b) < l {
					return nil, errMessageTooShort
				}
				b = b[l:]
			case sysAF_INET, sysAF_INET6:
				af = int(b[1])
				a, err := parseInetAddr(af, b)
				if err != nil {
					return nil, err
				}
				as[i] = a
				l := roundup(int(b[0]))
				if len(b) < l {
					return nil, errMessageTooShort
				}
				b = b[l:]
			default:
				l, a, err := fn(af, b)
				if err != nil {
					return nil, err
				}
				as[i] = a
				ll := roundup(l)
				if len(b) < ll {
					b = b[l:]
				} else {
					b = b[ll:]
				}
			}
		} else {
			a, err := parseDefaultAddr(b)
			if err != nil {
				return nil, err
			}
			as[i] = a
			l := roundup(int(b[0]))
			if len(b) < l {
				return nil, errMessageTooShort
			}
			b = b[l:]
		}
	}
	return as[:], nil
}

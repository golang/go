// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package route

import (
	"runtime"
	"syscall"
)

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
func (a *LinkAddr) Family() int { return syscall.AF_LINK }

func (a *LinkAddr) lenAndSpace() (int, int) {
	l := 8 + len(a.Name) + len(a.Addr)
	return l, roundup(l)
}

func (a *LinkAddr) marshal(b []byte) (int, error) {
	l, ll := a.lenAndSpace()
	if len(b) < ll {
		return 0, errShortBuffer
	}
	nlen, alen := len(a.Name), len(a.Addr)
	if nlen > 255 || alen > 255 {
		return 0, errInvalidAddr
	}
	b[0] = byte(l)
	b[1] = syscall.AF_LINK
	if a.Index > 0 {
		nativeEndian.PutUint16(b[2:4], uint16(a.Index))
	}
	data := b[8:]
	if nlen > 0 {
		b[5] = byte(nlen)
		copy(data[:nlen], a.Name)
		data = data[nlen:]
	}
	if alen > 0 {
		b[6] = byte(alen)
		copy(data[:alen], a.Addr)
		data = data[alen:]
	}
	return ll, nil
}

func parseLinkAddr(b []byte) (Addr, error) {
	if len(b) < 8 {
		return nil, errInvalidAddr
	}
	_, a, err := parseKernelLinkAddr(syscall.AF_LINK, b[4:])
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
func (a *Inet4Addr) Family() int { return syscall.AF_INET }

func (a *Inet4Addr) lenAndSpace() (int, int) {
	return sizeofSockaddrInet, roundup(sizeofSockaddrInet)
}

func (a *Inet4Addr) marshal(b []byte) (int, error) {
	l, ll := a.lenAndSpace()
	if len(b) < ll {
		return 0, errShortBuffer
	}
	b[0] = byte(l)
	b[1] = syscall.AF_INET
	copy(b[4:8], a.IP[:])
	return ll, nil
}

// An Inet6Addr represents an internet address for IPv6.
type Inet6Addr struct {
	IP     [16]byte // IP address
	ZoneID int      // zone identifier
}

// Family implements the Family method of Addr interface.
func (a *Inet6Addr) Family() int { return syscall.AF_INET6 }

func (a *Inet6Addr) lenAndSpace() (int, int) {
	return sizeofSockaddrInet6, roundup(sizeofSockaddrInet6)
}

func (a *Inet6Addr) marshal(b []byte) (int, error) {
	l, ll := a.lenAndSpace()
	if len(b) < ll {
		return 0, errShortBuffer
	}
	b[0] = byte(l)
	b[1] = syscall.AF_INET6
	copy(b[8:24], a.IP[:])
	if a.ZoneID > 0 {
		nativeEndian.PutUint32(b[24:28], uint32(a.ZoneID))
	}
	return ll, nil
}

// parseInetAddr parses b as an internet address for IPv4 or IPv6.
func parseInetAddr(af int, b []byte) (Addr, error) {
	const (
		off4 = 4 // offset of in_addr
		off6 = 8 // offset of in6_addr
	)
	switch af {
	case syscall.AF_INET:
		if len(b) < (off4+1) || len(b) < int(b[0]) || b[0] == 0 {
			return nil, errInvalidAddr
		}
		sockAddrLen := int(b[0])
		a := &Inet4Addr{}
		n := off4 + 4
		if sockAddrLen < n {
			n = sockAddrLen
		}
		copy(a.IP[:], b[off4:n])
		return a, nil
	case syscall.AF_INET6:
		if len(b) < (off6+1) || len(b) < int(b[0]) || b[0] == 0 {
			return nil, errInvalidAddr
		}
		sockAddrLen := int(b[0])
		n := off6 + 16
		if sockAddrLen < n {
			n = sockAddrLen
		}
		a := &Inet6Addr{}
		if sockAddrLen == sizeofSockaddrInet6 {
			a.ZoneID = int(nativeEndian.Uint32(b[24:28]))
		}
		copy(a.IP[:], b[off6:n])
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
	if runtime.GOOS == "darwin" || runtime.GOOS == "ios" {
		// On Darwin, an address in the kernel form is also
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
	case b[0] == sizeofSockaddrInet6:
		a := &Inet6Addr{}
		copy(a.IP[:], b[off6:off6+16])
		return int(b[0]), a, nil
	case af == syscall.AF_INET6:
		a := &Inet6Addr{}
		if l-1 < off6 {
			copy(a.IP[:], b[1:l])
		} else {
			copy(a.IP[:], b[l-off6:l])
		}
		return int(b[0]), a, nil
	case b[0] == sizeofSockaddrInet:
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

func (a *DefaultAddr) lenAndSpace() (int, int) {
	l := len(a.Raw)
	return l, roundup(l)
}

func (a *DefaultAddr) marshal(b []byte) (int, error) {
	l, ll := a.lenAndSpace()
	if len(b) < ll {
		return 0, errShortBuffer
	}
	if l > 255 {
		return 0, errInvalidAddr
	}
	b[1] = byte(l)
	copy(b[:l], a.Raw)
	return ll, nil
}

func parseDefaultAddr(b []byte) (Addr, error) {
	if len(b) < 2 || len(b) < int(b[0]) {
		return nil, errInvalidAddr
	}
	a := &DefaultAddr{af: int(b[1]), Raw: b[:b[0]]}
	return a, nil
}

func addrsSpace(as []Addr) int {
	var l int
	for _, a := range as {
		switch a := a.(type) {
		case *LinkAddr:
			_, ll := a.lenAndSpace()
			l += ll
		case *Inet4Addr:
			_, ll := a.lenAndSpace()
			l += ll
		case *Inet6Addr:
			_, ll := a.lenAndSpace()
			l += ll
		case *DefaultAddr:
			_, ll := a.lenAndSpace()
			l += ll
		}
	}
	return l
}

// marshalAddrs marshals as and returns a bitmap indicating which
// address is stored in b.
func marshalAddrs(b []byte, as []Addr) (uint, error) {
	var attrs uint
	for i, a := range as {
		switch a := a.(type) {
		case *LinkAddr:
			l, err := a.marshal(b)
			if err != nil {
				return 0, err
			}
			b = b[l:]
			attrs |= 1 << uint(i)
		case *Inet4Addr:
			l, err := a.marshal(b)
			if err != nil {
				return 0, err
			}
			b = b[l:]
			attrs |= 1 << uint(i)
		case *Inet6Addr:
			l, err := a.marshal(b)
			if err != nil {
				return 0, err
			}
			b = b[l:]
			attrs |= 1 << uint(i)
		case *DefaultAddr:
			l, err := a.marshal(b)
			if err != nil {
				return 0, err
			}
			b = b[l:]
			attrs |= 1 << uint(i)
		}
	}
	return attrs, nil
}

func parseAddrs(attrs uint, fn func(int, []byte) (int, Addr, error), b []byte) ([]Addr, error) {
	var as [syscall.RTAX_MAX]Addr
	af := int(syscall.AF_UNSPEC)
	for i := uint(0); i < syscall.RTAX_MAX && len(b) >= roundup(0); i++ {
		if attrs&(1<<i) == 0 {
			continue
		}
		if i <= syscall.RTAX_BRD {
			switch b[1] {
			case syscall.AF_LINK:
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
			case syscall.AF_INET, syscall.AF_INET6:
				// #70528: if the sockaddrlen is 0, no address to parse inside,
				// skip over the record.
				if b[0] > 0 {
					af = int(b[1])
					a, err := parseInetAddr(af, b)
					if err != nil {
						return nil, err
					}
					as[i] = a
				}
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
	// The only remaining bytes in b should be alignment.
	// However, under some circumstances DragonFly BSD appears to put
	// more addresses in the message than are indicated in the address
	// bitmask, so don't check for this.
	return as[:], nil
}

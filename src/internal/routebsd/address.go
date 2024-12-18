// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package routebsd

import (
	"net/netip"
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

// An InetAddr represent an internet address using IPv4 or IPv6.
type InetAddr struct {
	IP netip.Addr
}

func (a *InetAddr) Family() int {
	if a.IP.Is4() {
		return syscall.AF_INET
	} else {
		return syscall.AF_INET6
	}
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
		var ip [4]byte
		n := off4 + 4
		if sockAddrLen < n {
			n = sockAddrLen
		}
		copy(ip[:], b[off4:n])
		a := &InetAddr{
			IP: netip.AddrFrom4(ip),
		}
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
		var ip [16]byte
		copy(ip[:], b[off6:n])
		if ip[0] == 0xfe && ip[1]&0xc0 == 0x80 || ip[0] == 0xff && (ip[1]&0x0f == 0x01 || ip[1]&0x0f == 0x02) {
			// KAME based IPv6 protocol stack usually
			// embeds the interface index in the
			// interface-local or link-local address as
			// the kernel-internal form.
			id := int(bigEndian.Uint16(ip[2:4]))
			if id != 0 {
				ip[2], ip[3] = 0, 0
			}
		}
		// The kernel can provide an integer zone ID.
		// We ignore it.
		a := &InetAddr{
			IP: netip.AddrFrom16(ip),
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
	case b[0] == syscall.SizeofSockaddrInet6:
		a := &InetAddr{
			IP: netip.AddrFrom16([16]byte(b[off6:off6+16])),
		}
		return int(b[0]), a, nil
	case af == syscall.AF_INET6:
		var ab[16]byte
		if l-1 < off6 {
			copy(ab[:], b[1:l])
		} else {
			copy(ab[:], b[l-off6:l])
		}
		a := &InetAddr{
			IP: netip.AddrFrom16(ab),
		}
		return int(b[0]), a, nil
	case b[0] == syscall.SizeofSockaddrInet4:
		a := &InetAddr{
			IP: netip.AddrFrom4([4]byte(b[off4:off4+4])),
		}
		return int(b[0]), a, nil
	default: // an old fashion, AF_UNSPEC or unknown means AF_INET
		var ab [4]byte
		if l-1 < off4 {
			copy(ab[:], b[1:l])
		} else {
			copy(ab[:], b[l-off4:l])
		}
		a := &InetAddr{
			IP: netip.AddrFrom4(ab),
		}
		return int(b[0]), a, nil
	}
}

func parseAddrs(attrs uint, b []byte) ([]Addr, error) {
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
				l, a, err := parseKernelInetAddr(af, b)
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
			// Skip unknown addresses.
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

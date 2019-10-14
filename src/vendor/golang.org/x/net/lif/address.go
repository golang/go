// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris

package lif

import (
	"errors"
	"unsafe"
)

// An Addr represents an address associated with packet routing.
type Addr interface {
	// Family returns an address family.
	Family() int
}

// An Inet4Addr represents an internet address for IPv4.
type Inet4Addr struct {
	IP        [4]byte // IP address
	PrefixLen int     // address prefix length
}

// Family implements the Family method of Addr interface.
func (a *Inet4Addr) Family() int { return sysAF_INET }

// An Inet6Addr represents an internet address for IPv6.
type Inet6Addr struct {
	IP        [16]byte // IP address
	PrefixLen int      // address prefix length
	ZoneID    int      // zone identifier
}

// Family implements the Family method of Addr interface.
func (a *Inet6Addr) Family() int { return sysAF_INET6 }

// Addrs returns a list of interface addresses.
//
// The provided af must be an address family and name must be a data
// link name. The zero value of af or name means a wildcard.
func Addrs(af int, name string) ([]Addr, error) {
	eps, err := newEndpoints(af)
	if len(eps) == 0 {
		return nil, err
	}
	defer func() {
		for _, ep := range eps {
			ep.close()
		}
	}()
	lls, err := links(eps, name)
	if len(lls) == 0 {
		return nil, err
	}
	var as []Addr
	for _, ll := range lls {
		var lifr lifreq
		for i := 0; i < len(ll.Name); i++ {
			lifr.Name[i] = int8(ll.Name[i])
		}
		for _, ep := range eps {
			ioc := int64(sysSIOCGLIFADDR)
			err := ioctl(ep.s, uintptr(ioc), unsafe.Pointer(&lifr))
			if err != nil {
				continue
			}
			sa := (*sockaddrStorage)(unsafe.Pointer(&lifr.Lifru[0]))
			l := int(nativeEndian.Uint32(lifr.Lifru1[:4]))
			if l == 0 {
				continue
			}
			switch sa.Family {
			case sysAF_INET:
				a := &Inet4Addr{PrefixLen: l}
				copy(a.IP[:], lifr.Lifru[4:8])
				as = append(as, a)
			case sysAF_INET6:
				a := &Inet6Addr{PrefixLen: l, ZoneID: int(nativeEndian.Uint32(lifr.Lifru[24:28]))}
				copy(a.IP[:], lifr.Lifru[8:24])
				as = append(as, a)
			}
		}
	}
	return as, nil
}

func parseLinkAddr(b []byte) ([]byte, error) {
	nlen, alen, slen := int(b[1]), int(b[2]), int(b[3])
	l := 4 + nlen + alen + slen
	if len(b) < l {
		return nil, errors.New("invalid address")
	}
	b = b[4:]
	var addr []byte
	if nlen > 0 {
		b = b[nlen:]
	}
	if alen > 0 {
		addr = make([]byte, alen)
		copy(addr, b[:alen])
	}
	return addr, nil
}

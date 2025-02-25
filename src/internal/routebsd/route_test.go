// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package routebsd

import (
	"fmt"
	"runtime"
	"syscall"
)

func (m *InterfaceMessage) String() string {
	var attrs addrAttrs
	if runtime.GOOS == "openbsd" {
		attrs = addrAttrs(nativeEndian.Uint32(m.raw[12:16]))
	} else {
		attrs = addrAttrs(nativeEndian.Uint32(m.raw[4:8]))
	}
	return fmt.Sprintf("%s", attrs)
}

func (m *InterfaceAddrMessage) String() string {
	var attrs addrAttrs
	if runtime.GOOS == "openbsd" {
		attrs = addrAttrs(nativeEndian.Uint32(m.raw[12:16]))
	} else {
		attrs = addrAttrs(nativeEndian.Uint32(m.raw[4:8]))
	}
	return fmt.Sprintf("%s", attrs)
}

func (m *InterfaceMulticastAddrMessage) String() string {
	return fmt.Sprintf("%s", addrAttrs(nativeEndian.Uint32(m.raw[4:8])))
}

type addrAttrs uint

var addrAttrNames = [...]string{
	"dst",
	"gateway",
	"netmask",
	"genmask",
	"ifp",
	"ifa",
	"author",
	"brd",
	"df:mpls1-n:tag-o:src", // mpls1 for dragonfly, tag for netbsd, src for openbsd
	"df:mpls2-o:srcmask",   // mpls2 for dragonfly, srcmask for openbsd
	"df:mpls3-o:label",     // mpls3 for dragonfly, label for openbsd
	"o:bfd",                // bfd for openbsd
	"o:dns",                // dns for openbsd
	"o:static",             // static for openbsd
	"o:search",             // search for openbsd
}

func (attrs addrAttrs) String() string {
	var s string
	for i, name := range addrAttrNames {
		if attrs&(1<<uint(i)) != 0 {
			if s != "" {
				s += "|"
			}
			s += name
		}
	}
	if s == "" {
		return "<nil>"
	}
	return s
}

type msgs []Message

func (ms msgs) validate() ([]string, error) {
	var ss []string
	for _, m := range ms {
		switch m := m.(type) {
		case *InterfaceMessage:
			var attrs addrAttrs
			if runtime.GOOS == "openbsd" {
				attrs = addrAttrs(nativeEndian.Uint32(m.raw[12:16]))
			} else {
				attrs = addrAttrs(nativeEndian.Uint32(m.raw[4:8]))
			}
			if err := addrs(m.Addrs).match(attrs); err != nil {
				return nil, err
			}
			ss = append(ss, m.String()+" "+addrs(m.Addrs).String())
		case *InterfaceAddrMessage:
			var attrs addrAttrs
			if runtime.GOOS == "openbsd" {
				attrs = addrAttrs(nativeEndian.Uint32(m.raw[12:16]))
			} else {
				attrs = addrAttrs(nativeEndian.Uint32(m.raw[4:8]))
			}
			if err := addrs(m.Addrs).match(attrs); err != nil {
				return nil, err
			}
			ss = append(ss, m.String()+" "+addrs(m.Addrs).String())
		case *InterfaceMulticastAddrMessage:
			if err := addrs(m.Addrs).match(addrAttrs(nativeEndian.Uint32(m.raw[4:8]))); err != nil {
				return nil, err
			}
			ss = append(ss, m.String()+" "+addrs(m.Addrs).String())
		default:
			ss = append(ss, fmt.Sprintf("%+v", m))
		}
	}
	return ss, nil
}

type addrFamily int

func (af addrFamily) String() string {
	switch af {
	case syscall.AF_UNSPEC:
		return "unspec"
	case syscall.AF_LINK:
		return "link"
	case syscall.AF_INET:
		return "inet4"
	case syscall.AF_INET6:
		return "inet6"
	default:
		return fmt.Sprintf("%d", af)
	}
}

const hexDigit = "0123456789abcdef"

type llAddr []byte

func (a llAddr) String() string {
	if len(a) == 0 {
		return ""
	}
	buf := make([]byte, 0, len(a)*3-1)
	for i, b := range a {
		if i > 0 {
			buf = append(buf, ':')
		}
		buf = append(buf, hexDigit[b>>4])
		buf = append(buf, hexDigit[b&0xF])
	}
	return string(buf)
}

type ipAddr []byte

func (a ipAddr) String() string {
	if len(a) == 0 {
		return "<nil>"
	}
	if len(a) == 4 {
		return fmt.Sprintf("%d.%d.%d.%d", a[0], a[1], a[2], a[3])
	}
	if len(a) == 16 {
		return fmt.Sprintf("%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x", a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8], a[9], a[10], a[11], a[12], a[13], a[14], a[15])
	}
	s := make([]byte, len(a)*2)
	for i, tn := range a {
		s[i*2], s[i*2+1] = hexDigit[tn>>4], hexDigit[tn&0xf]
	}
	return string(s)
}

func (a *LinkAddr) String() string {
	name := a.Name
	if name == "" {
		name = "<nil>"
	}
	lla := llAddr(a.Addr).String()
	if lla == "" {
		lla = "<nil>"
	}
	return fmt.Sprintf("(%v %d %s %s)", addrFamily(a.Family()), a.Index, name, lla)
}

func (a *InetAddr) String() string {
	return fmt.Sprintf("(%v %v)", addrFamily(a.Family()), a.IP)
}

type addrs []Addr

func (as addrs) String() string {
	var s string
	for _, a := range as {
		if a == nil {
			continue
		}
		if len(s) > 0 {
			s += " "
		}
		switch a := a.(type) {
		case *LinkAddr:
			s += a.String()
		case *InetAddr:
			s += a.String()
		}
	}
	if s == "" {
		return "<nil>"
	}
	return s
}

func (as addrs) match(attrs addrAttrs) error {
	var ts addrAttrs
	af := syscall.AF_UNSPEC
	for i := range as {
		if as[i] != nil {
			ts |= 1 << uint(i)
		}
		switch addr := as[i].(type) {
		case *InetAddr:
			got := 0
			if addr.IP.Is4() {
				got = syscall.AF_INET
			} else if addr.IP.Is6() {
				got = syscall.AF_INET6
			}
			if af == syscall.AF_UNSPEC {
				if got != 0 {
					af = got
				}
			}
			if got != 0 && af != got {
				return fmt.Errorf("got %v; want %v", addrs(as), addrFamily(af))
			}
		}
	}
	if ts != attrs && ts > attrs {
		return fmt.Errorf("%v not included in %v", ts, attrs)
	}
	return nil
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build darwin dragonfly freebsd netbsd openbsd

package route

import (
	"fmt"
	"os/exec"
	"runtime"
	"time"
)

func (m *RouteMessage) String() string {
	return fmt.Sprintf("%s", addrAttrs(nativeEndian.Uint32(m.raw[12:16])))
}

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

func (m *InterfaceAnnounceMessage) String() string {
	what := "<nil>"
	switch m.What {
	case 0:
		what = "arrival"
	case 1:
		what = "departure"
	}
	return fmt.Sprintf("(%d %s %s)", m.Index, m.Name, what)
}

func (m *InterfaceMetrics) String() string {
	return fmt.Sprintf("(type=%d mtu=%d)", m.Type, m.MTU)
}

func (m *RouteMetrics) String() string {
	return fmt.Sprintf("(pmtu=%d)", m.PathMTU)
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
		case *RouteMessage:
			if err := addrs(m.Addrs).match(addrAttrs(nativeEndian.Uint32(m.raw[12:16]))); err != nil {
				return nil, err
			}
			sys := m.Sys()
			if sys == nil {
				return nil, fmt.Errorf("no sys for %s", m.String())
			}
			ss = append(ss, m.String()+" "+syss(sys).String()+" "+addrs(m.Addrs).String())
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
			sys := m.Sys()
			if sys == nil {
				return nil, fmt.Errorf("no sys for %s", m.String())
			}
			ss = append(ss, m.String()+" "+syss(sys).String()+" "+addrs(m.Addrs).String())
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
		case *InterfaceAnnounceMessage:
			ss = append(ss, m.String())
		default:
			ss = append(ss, fmt.Sprintf("%+v", m))
		}
	}
	return ss, nil
}

type syss []Sys

func (sys syss) String() string {
	var s string
	for _, sy := range sys {
		switch sy := sy.(type) {
		case *InterfaceMetrics:
			if len(s) > 0 {
				s += " "
			}
			s += sy.String()
		case *RouteMetrics:
			if len(s) > 0 {
				s += " "
			}
			s += sy.String()
		}
	}
	return s
}

type addrFamily int

func (af addrFamily) String() string {
	switch af {
	case sysAF_UNSPEC:
		return "unspec"
	case sysAF_LINK:
		return "link"
	case sysAF_INET:
		return "inet4"
	case sysAF_INET6:
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

func (a *Inet4Addr) String() string {
	return fmt.Sprintf("(%v %v)", addrFamily(a.Family()), ipAddr(a.IP[:]))
}

func (a *Inet6Addr) String() string {
	return fmt.Sprintf("(%v %v %d)", addrFamily(a.Family()), ipAddr(a.IP[:]), a.ZoneID)
}

func (a *DefaultAddr) String() string {
	return fmt.Sprintf("(%v %s)", addrFamily(a.Family()), ipAddr(a.Raw[2:]).String())
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
		case *Inet4Addr:
			s += a.String()
		case *Inet6Addr:
			s += a.String()
		case *DefaultAddr:
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
	af := sysAF_UNSPEC
	for i := range as {
		if as[i] != nil {
			ts |= 1 << uint(i)
		}
		switch as[i].(type) {
		case *Inet4Addr:
			if af == sysAF_UNSPEC {
				af = sysAF_INET
			}
			if af != sysAF_INET {
				return fmt.Errorf("got %v; want %v", addrs(as), addrFamily(af))
			}
		case *Inet6Addr:
			if af == sysAF_UNSPEC {
				af = sysAF_INET6
			}
			if af != sysAF_INET6 {
				return fmt.Errorf("got %v; want %v", addrs(as), addrFamily(af))
			}
		}
	}
	if ts != attrs && ts > attrs {
		return fmt.Errorf("%v not included in %v", ts, attrs)
	}
	return nil
}

func fetchAndParseRIB(af int, typ RIBType) ([]Message, error) {
	var err error
	var b []byte
	for i := 0; i < 3; i++ {
		if b, err = FetchRIB(af, typ, 0); err != nil {
			time.Sleep(10 * time.Millisecond)
			continue
		}
		break
	}
	if err != nil {
		return nil, fmt.Errorf("%v %d %v", addrFamily(af), typ, err)
	}
	ms, err := ParseRIB(typ, b)
	if err != nil {
		return nil, fmt.Errorf("%v %d %v", addrFamily(af), typ, err)
	}
	return ms, nil
}

// propVirtual is a proprietary virtual network interface.
type propVirtual struct {
	name         string
	addr, mask   string
	setupCmds    []*exec.Cmd
	teardownCmds []*exec.Cmd
}

func (pv *propVirtual) setup() error {
	for _, cmd := range pv.setupCmds {
		if err := cmd.Run(); err != nil {
			pv.teardown()
			return err
		}
	}
	return nil
}

func (pv *propVirtual) teardown() error {
	for _, cmd := range pv.teardownCmds {
		if err := cmd.Run(); err != nil {
			return err
		}
	}
	return nil
}

func (pv *propVirtual) configure(suffix int) error {
	if runtime.GOOS == "openbsd" {
		pv.name = fmt.Sprintf("vether%d", suffix)
	} else {
		pv.name = fmt.Sprintf("vlan%d", suffix)
	}
	xname, err := exec.LookPath("ifconfig")
	if err != nil {
		return err
	}
	pv.setupCmds = append(pv.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", pv.name, "create"},
	})
	if runtime.GOOS == "netbsd" {
		// NetBSD requires an underlying dot1Q-capable network
		// interface.
		pv.setupCmds = append(pv.setupCmds, &exec.Cmd{
			Path: xname,
			Args: []string{"ifconfig", pv.name, "vlan", fmt.Sprintf("%d", suffix&0xfff), "vlanif", "wm0"},
		})
	}
	pv.setupCmds = append(pv.setupCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", pv.name, "inet", pv.addr, "netmask", pv.mask},
	})
	pv.teardownCmds = append(pv.teardownCmds, &exec.Cmd{
		Path: xname,
		Args: []string{"ifconfig", pv.name, "destroy"},
	})
	return nil
}

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build solaris

package lif

import (
	"fmt"
	"testing"
)

type addrFamily int

func (af addrFamily) String() string {
	switch af {
	case sysAF_UNSPEC:
		return "unspec"
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

func (a *Inet4Addr) String() string {
	return fmt.Sprintf("(%s %s %d)", addrFamily(a.Family()), ipAddr(a.IP[:]), a.PrefixLen)
}

func (a *Inet6Addr) String() string {
	return fmt.Sprintf("(%s %s %d %d)", addrFamily(a.Family()), ipAddr(a.IP[:]), a.PrefixLen, a.ZoneID)
}

type addrPack struct {
	af int
	as []Addr
}

func addrPacks() ([]addrPack, error) {
	var lastErr error
	var aps []addrPack
	for _, af := range [...]int{sysAF_UNSPEC, sysAF_INET, sysAF_INET6} {
		as, err := Addrs(af, "")
		if err != nil {
			lastErr = err
			continue
		}
		aps = append(aps, addrPack{af: af, as: as})
	}
	return aps, lastErr
}

func TestAddrs(t *testing.T) {
	aps, err := addrPacks()
	if len(aps) == 0 && err != nil {
		t.Fatal(err)
	}
	lps, err := linkPacks()
	if len(lps) == 0 && err != nil {
		t.Fatal(err)
	}
	for _, lp := range lps {
		n := 0
		for _, ll := range lp.lls {
			as, err := Addrs(lp.af, ll.Name)
			if err != nil {
				t.Fatal(lp.af, ll.Name, err)
			}
			t.Logf("af=%s name=%s %v", addrFamily(lp.af), ll.Name, as)
			n += len(as)
		}
		for _, ap := range aps {
			if ap.af != lp.af {
				continue
			}
			if n != len(ap.as) {
				t.Errorf("af=%s got %d; want %d", addrFamily(lp.af), n, len(ap.as))
				continue
			}
		}
	}
}

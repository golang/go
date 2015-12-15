// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"internal/syscall/windows"
	"sort"
	"testing"
)

func TestWindowsInterfaces(t *testing.T) {
	aas, err := adapterAddresses()
	if err != nil {
		t.Fatal(err)
	}
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	for i, ifi := range ift {
		aa := aas[i]
		if len(ifi.HardwareAddr) != int(aa.PhysicalAddressLength) {
			t.Errorf("got %d; want %d", len(ifi.HardwareAddr), aa.PhysicalAddressLength)
		}
		if ifi.MTU > 0x7fffffff {
			t.Errorf("%s: got %d; want less than or equal to 1<<31 - 1", ifi.Name, ifi.MTU)
		}
		if ifi.Flags&FlagUp != 0 && aa.OperStatus != windows.IfOperStatusUp {
			t.Errorf("%s: got %v; should not include FlagUp", ifi.Name, ifi.Flags)
		}
		if ifi.Flags&FlagLoopback != 0 && aa.IfType != windows.IF_TYPE_SOFTWARE_LOOPBACK {
			t.Errorf("%s: got %v; should not include FlagLoopback", ifi.Name, ifi.Flags)
		}
		if _, _, err := addrPrefixTable(aa); err != nil {
			t.Errorf("%s: %v", ifi.Name, err)
		}
	}
}

type byAddrLen []IPNet

func (ps byAddrLen) Len() int { return len(ps) }

func (ps byAddrLen) Less(i, j int) bool {
	if n := bytes.Compare(ps[i].IP, ps[j].IP); n != 0 {
		return n < 0
	}
	if n := bytes.Compare(ps[i].Mask, ps[j].Mask); n != 0 {
		return n < 0
	}
	return false
}

func (ps byAddrLen) Swap(i, j int) { ps[i], ps[j] = ps[j], ps[i] }

var windowsAddrPrefixLenTests = []struct {
	pfxs []IPNet
	ip   IP
	out  int
}{
	{
		[]IPNet{
			{IP: IPv4(172, 16, 0, 0), Mask: IPv4Mask(255, 255, 0, 0)},
			{IP: IPv4(192, 168, 0, 0), Mask: IPv4Mask(255, 255, 255, 0)},
			{IP: IPv4(192, 168, 0, 0), Mask: IPv4Mask(255, 255, 255, 128)},
			{IP: IPv4(192, 168, 0, 0), Mask: IPv4Mask(255, 255, 255, 192)},
		},
		IPv4(192, 168, 0, 1),
		26,
	},
	{
		[]IPNet{
			{IP: ParseIP("2001:db8::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff0"))},
			{IP: ParseIP("2001:db8::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff8"))},
			{IP: ParseIP("2001:db8::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffc"))},
		},
		ParseIP("2001:db8::1"),
		126,
	},

	// Fallback cases. It may happen on Windows XP or 2003 server.
	{
		[]IPNet{
			{IP: IPv4(127, 0, 0, 0).To4(), Mask: IPv4Mask(255, 0, 0, 0)},
			{IP: IPv4(10, 0, 0, 0).To4(), Mask: IPv4Mask(255, 0, 0, 0)},
			{IP: IPv4(172, 16, 0, 0).To4(), Mask: IPv4Mask(255, 255, 0, 0)},
			{IP: IPv4(192, 168, 255, 0), Mask: IPv4Mask(255, 255, 255, 0)},
			{IP: IPv4zero, Mask: IPv4Mask(0, 0, 0, 0)},
		},
		IPv4(192, 168, 0, 1),
		8 * IPv4len,
	},
	{
		nil,
		IPv4(192, 168, 0, 1),
		8 * IPv4len,
	},
	{
		[]IPNet{
			{IP: IPv6loopback, Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff"))},
			{IP: ParseIP("2001:db8:1::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff0"))},
			{IP: ParseIP("2001:db8:2::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fff8"))},
			{IP: ParseIP("2001:db8:3::"), Mask: IPMask(ParseIP("ffff:ffff:ffff:ffff:ffff:ffff:ffff:fffc"))},
			{IP: IPv6unspecified, Mask: IPMask(ParseIP("::"))},
		},
		ParseIP("2001:db8::1"),
		8 * IPv6len,
	},
	{
		nil,
		ParseIP("2001:db8::1"),
		8 * IPv6len,
	},
}

func TestWindowsAddrPrefixLen(t *testing.T) {
	for i, tt := range windowsAddrPrefixLenTests {
		sort.Sort(byAddrLen(tt.pfxs))
		l := addrPrefixLen(tt.pfxs, tt.ip)
		if l != tt.out {
			t.Errorf("#%d: got %d; want %d", i, l, tt.out)
		}
		sort.Sort(sort.Reverse(byAddrLen(tt.pfxs)))
		l = addrPrefixLen(tt.pfxs, tt.ip)
		if l != tt.out {
			t.Errorf("#%d: got %d; want %d", i, l, tt.out)
		}
	}
}

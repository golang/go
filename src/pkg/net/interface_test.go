// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

func sameInterface(i, j *Interface) bool {
	if i == nil || j == nil {
		return false
	}
	if i.Index == j.Index && i.Name == j.Name && bytes.Equal(i.HardwareAddr, j.HardwareAddr) {
		return true
	}
	return false
}

func TestInterfaces(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatalf("Interfaces failed: %v", err)
	}
	t.Logf("table: len/cap = %v/%v\n", len(ift), cap(ift))

	for _, ifi := range ift {
		ifxi, err := InterfaceByIndex(ifi.Index)
		if err != nil {
			t.Fatalf("InterfaceByIndex(%#q) failed: %v", ifi.Index, err)
		}
		if !sameInterface(ifxi, &ifi) {
			t.Fatalf("InterfaceByIndex(%#q) = %v, want %v", ifi.Index, *ifxi, ifi)
		}
		ifxn, err := InterfaceByName(ifi.Name)
		if err != nil {
			t.Fatalf("InterfaceByName(%#q) failed: %v", ifi.Name, err)
		}
		if !sameInterface(ifxn, &ifi) {
			t.Fatalf("InterfaceByName(%#q) = %v, want %v", ifi.Name, *ifxn, ifi)
		}
		t.Logf("%q: flags %q, ifindex %v, mtu %v\n", ifi.Name, ifi.Flags.String(), ifi.Index, ifi.MTU)
		t.Logf("\thardware address %q", ifi.HardwareAddr.String())
		testInterfaceAddrs(t, &ifi)
		testInterfaceMulticastAddrs(t, &ifi)
	}
}

func TestInterfaceAddrs(t *testing.T) {
	ifat, err := InterfaceAddrs()
	if err != nil {
		t.Fatalf("InterfaceAddrs failed: %v", err)
	}
	t.Logf("table: len/cap = %v/%v\n", len(ifat), cap(ifat))
	testAddrs(t, ifat)
}

func testInterfaceAddrs(t *testing.T, ifi *Interface) {
	ifat, err := ifi.Addrs()
	if err != nil {
		t.Fatalf("Interface.Addrs failed: %v", err)
	}
	testAddrs(t, ifat)
}

func testInterfaceMulticastAddrs(t *testing.T, ifi *Interface) {
	ifmat, err := ifi.MulticastAddrs()
	if err != nil {
		t.Fatalf("Interface.MulticastAddrs failed: %v", err)
	}
	testMulticastAddrs(t, ifmat)
}

func testAddrs(t *testing.T, ifat []Addr) {
	for _, ifa := range ifat {
		switch ifa.(type) {
		case *IPAddr, *IPNet:
			t.Logf("\tinterface address %q\n", ifa.String())
		default:
			t.Errorf("\tunexpected type: %T", ifa)
		}
	}
}

func testMulticastAddrs(t *testing.T, ifmat []Addr) {
	for _, ifma := range ifmat {
		switch ifma.(type) {
		case *IPAddr:
			t.Logf("\tjoined group address %q\n", ifma.String())
		default:
			t.Errorf("\tunexpected type: %T", ifma)
		}
	}
}

var mactests = []struct {
	in  string
	out HardwareAddr
	err string
}{
	{"01:23:45:67:89:AB", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab}, ""},
	{"01-23-45-67-89-AB", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab}, ""},
	{"0123.4567.89AB", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab}, ""},
	{"ab:cd:ef:AB:CD:EF", HardwareAddr{0xab, 0xcd, 0xef, 0xab, 0xcd, 0xef}, ""},
	{"01.02.03.04.05.06", nil, "invalid MAC address"},
	{"01:02:03:04:05:06:", nil, "invalid MAC address"},
	{"x1:02:03:04:05:06", nil, "invalid MAC address"},
	{"01002:03:04:05:06", nil, "invalid MAC address"},
	{"01:02003:04:05:06", nil, "invalid MAC address"},
	{"01:02:03004:05:06", nil, "invalid MAC address"},
	{"01:02:03:04005:06", nil, "invalid MAC address"},
	{"01:02:03:04:05006", nil, "invalid MAC address"},
	{"01-02:03:04:05:06", nil, "invalid MAC address"},
	{"01:02-03-04-05-06", nil, "invalid MAC address"},
	{"0123:4567:89AF", nil, "invalid MAC address"},
	{"0123-4567-89AF", nil, "invalid MAC address"},
	{"01:23:45:67:89:AB:CD:EF", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, ""},
	{"01-23-45-67-89-AB-CD-EF", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, ""},
	{"0123.4567.89AB.CDEF", HardwareAddr{1, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, ""},
}

func match(err error, s string) bool {
	if s == "" {
		return err == nil
	}
	return err != nil && strings.Contains(err.Error(), s)
}

func TestParseMAC(t *testing.T) {
	for _, tt := range mactests {
		out, err := ParseMAC(tt.in)
		if !reflect.DeepEqual(out, tt.out) || !match(err, tt.err) {
			t.Errorf("ParseMAC(%q) = %v, %v, want %v, %v", tt.in, out, err, tt.out,
				tt.err)
		}
	}
}

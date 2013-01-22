// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"bytes"
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
	t.Logf("table: len/cap = %v/%v", len(ift), cap(ift))

	for _, ifi := range ift {
		ifxi, err := InterfaceByIndex(ifi.Index)
		if err != nil {
			t.Fatalf("InterfaceByIndex(%q) failed: %v", ifi.Index, err)
		}
		if !sameInterface(ifxi, &ifi) {
			t.Fatalf("InterfaceByIndex(%q) = %v, want %v", ifi.Index, *ifxi, ifi)
		}
		ifxn, err := InterfaceByName(ifi.Name)
		if err != nil {
			t.Fatalf("InterfaceByName(%q) failed: %v", ifi.Name, err)
		}
		if !sameInterface(ifxn, &ifi) {
			t.Fatalf("InterfaceByName(%q) = %v, want %v", ifi.Name, *ifxn, ifi)
		}
		t.Logf("%q: flags %q, ifindex %v, mtu %v", ifi.Name, ifi.Flags.String(), ifi.Index, ifi.MTU)
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
	t.Logf("table: len/cap = %v/%v", len(ifat), cap(ifat))
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
		switch v := ifa.(type) {
		case *IPAddr, *IPNet:
			if v == nil {
				t.Errorf("\tunexpected value: %v", ifa)
			} else {
				t.Logf("\tinterface address %q", ifa.String())
			}
		default:
			t.Errorf("\tunexpected type: %T", ifa)
		}
	}
}

func testMulticastAddrs(t *testing.T, ifmat []Addr) {
	for _, ifma := range ifmat {
		switch v := ifma.(type) {
		case *IPAddr:
			if v == nil {
				t.Errorf("\tunexpected value: %v", ifma)
			} else {
				t.Logf("\tjoined group address %q", ifma.String())
			}
		default:
			t.Errorf("\tunexpected type: %T", ifma)
		}
	}
}

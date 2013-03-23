// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"testing"
)

// loopbackInterface returns an available logical network interface
// for loopback tests.  It returns nil if no suitable interface is
// found.
func loopbackInterface() *Interface {
	ift, err := Interfaces()
	if err != nil {
		return nil
	}
	for _, ifi := range ift {
		if ifi.Flags&FlagLoopback != 0 && ifi.Flags&FlagUp != 0 {
			return &ifi
		}
	}
	return nil
}

// ipv6LinkLocalUnicastAddr returns an IPv6 link-local unicast address
// on the given network interface for tests. It returns "" if no
// suitable address is found.
func ipv6LinkLocalUnicastAddr(ifi *Interface) string {
	if ifi == nil {
		return ""
	}
	ifat, err := ifi.Addrs()
	if err != nil {
		return ""
	}
	for _, ifa := range ifat {
		switch ifa := ifa.(type) {
		case *IPAddr:
			if ifa.IP.To4() == nil && ifa.IP.IsLinkLocalUnicast() {
				return ifa.IP.String()
			}
		case *IPNet:
			if ifa.IP.To4() == nil && ifa.IP.IsLinkLocalUnicast() {
				return ifa.IP.String()
			}
		}
	}
	return ""
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
			t.Fatalf("InterfaceByIndex(%v) failed: %v", ifi.Index, err)
		}
		if !reflect.DeepEqual(ifxi, &ifi) {
			t.Fatalf("InterfaceByIndex(%v) = %v, want %v", ifi.Index, ifxi, ifi)
		}
		ifxn, err := InterfaceByName(ifi.Name)
		if err != nil {
			t.Fatalf("InterfaceByName(%q) failed: %v", ifi.Name, err)
		}
		if !reflect.DeepEqual(ifxn, &ifi) {
			t.Fatalf("InterfaceByName(%q) = %v, want %v", ifi.Name, ifxn, ifi)
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
		switch ifa := ifa.(type) {
		case *IPAddr, *IPNet:
			if ifa == nil {
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
		switch ifma := ifma.(type) {
		case *IPAddr:
			if ifma == nil {
				t.Errorf("\tunexpected value: %v", ifma)
			} else {
				t.Logf("\tjoined group address %q", ifma.String())
			}
		default:
			t.Errorf("\tunexpected type: %T", ifma)
		}
	}
}

func BenchmarkInterfaces(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := Interfaces(); err != nil {
			b.Fatalf("Interfaces failed: %v", err)
		}
	}
}

func BenchmarkInterfaceByIndex(b *testing.B) {
	ifi := loopbackInterface()
	if ifi == nil {
		b.Skip("loopback interface not found")
	}
	for i := 0; i < b.N; i++ {
		if _, err := InterfaceByIndex(ifi.Index); err != nil {
			b.Fatalf("InterfaceByIndex failed: %v", err)
		}
	}
}

func BenchmarkInterfaceByName(b *testing.B) {
	ifi := loopbackInterface()
	if ifi == nil {
		b.Skip("loopback interface not found")
	}
	for i := 0; i < b.N; i++ {
		if _, err := InterfaceByName(ifi.Name); err != nil {
			b.Fatalf("InterfaceByName failed: %v", err)
		}
	}
}

func BenchmarkInterfaceAddrs(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := InterfaceAddrs(); err != nil {
			b.Fatalf("InterfaceAddrs failed: %v", err)
		}
	}
}

func BenchmarkInterfacesAndAddrs(b *testing.B) {
	ifi := loopbackInterface()
	if ifi == nil {
		b.Skip("loopback interface not found")
	}
	for i := 0; i < b.N; i++ {
		if _, err := ifi.Addrs(); err != nil {
			b.Fatalf("Interface.Addrs failed: %v", err)
		}
	}
}

func BenchmarkInterfacesAndMulticastAddrs(b *testing.B) {
	ifi := loopbackInterface()
	if ifi == nil {
		b.Skip("loopback interface not found")
	}
	for i := 0; i < b.N; i++ {
		if _, err := ifi.MulticastAddrs(); err != nil {
			b.Fatalf("Interface.MulticastAddrs failed: %v", err)
		}
	}
}

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"reflect"
	"runtime"
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
		case *IPNet:
			if ifa.IP.To4() == nil && ifa.IP.IsLinkLocalUnicast() {
				return ifa.IP.String()
			}
		}
	}
	return ""
}

func TestInterfaces(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("temporarily disabled until golang.org/issue/5395 is fixed")
	}

	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	var nifs, naf4, naf6, nmaf4, nmaf6 int
	for _, ifi := range ift {
		ifxi, err := InterfaceByIndex(ifi.Index)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ifxi, &ifi) {
			t.Errorf("got %v; want %v", ifxi, ifi)
		}
		ifxn, err := InterfaceByName(ifi.Name)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(ifxn, &ifi) {
			t.Errorf("got %v; want %v", ifxn, ifi)
		}
		t.Logf("%q: flags %q, ifindex %v, mtu %v", ifi.Name, ifi.Flags.String(), ifi.Index, ifi.MTU)
		t.Logf("hardware address %q", ifi.HardwareAddr.String())
		if ifi.Flags&FlagUp != 0 && ifi.Flags&FlagLoopback == 0 {
			nifs++ // active interfaces except loopback interfaces
		}
		n4, n6 := testInterfaceAddrs(t, &ifi)
		naf4 += n4
		naf6 += n6
		n4, n6 = testInterfaceMulticastAddrs(t, &ifi)
		nmaf4 += n4
		nmaf6 += n6
	}
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris":
	default:
		if supportsIPv4 && nifs > 0 && naf4 == 0 {
			t.Errorf("got %v; want more than or equal to one", naf4)
		}
		if supportsIPv6 && nifs > 0 && naf6 == 0 {
			t.Errorf("got %v; want more than or equal to one", naf6)
		}
	}
	switch runtime.GOOS {
	case "dragonfly", "nacl", "netbsd", "openbsd", "plan9", "solaris":
	default:
		// Unlike IPv6, IPv4 multicast capability is not a
		// mandatory feature.
		//if supportsIPv4 && nactvifs > 0 && nmaf4 == 0 {
		//	t.Errorf("got %v; want more than or equal to one", nmaf4)
		//}
		if supportsIPv6 && nifs > 0 && nmaf6 == 0 {
			t.Errorf("got %v; want more than or equal to one", nmaf6)
		}
	}
}

func TestInterfaceAddrs(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("temporarily disabled until golang.org/issue/5395 is fixed")
	}

	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	var nifs int
	for _, ifi := range ift {
		if ifi.Flags&FlagUp != 0 && ifi.Flags&FlagLoopback == 0 {
			nifs++ // active interfaces except loopback interfaces
		}
	}
	ifat, err := InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	naf4, naf6 := testAddrs(t, ifat)
	if supportsIPv4 && nifs > 0 && naf4 == 0 {
		t.Errorf("got %v; want more than or equal to one", naf4)
	}
	if supportsIPv6 && nifs > 0 && naf6 == 0 {
		t.Errorf("got %v; want more than or equal to one", naf6)
	}
}

func testInterfaceAddrs(t *testing.T, ifi *Interface) (naf4, naf6 int) {
	ifat, err := ifi.Addrs()
	if err != nil {
		t.Fatal(err)
	}
	return testAddrs(t, ifat)
}

func testInterfaceMulticastAddrs(t *testing.T, ifi *Interface) (nmaf4, nmaf6 int) {
	ifmat, err := ifi.MulticastAddrs()
	if err != nil {
		t.Fatal(err)
	}
	return testMulticastAddrs(t, ifmat)
}

func testAddrs(t *testing.T, ifat []Addr) (naf4, naf6 int) {
	for _, ifa := range ifat {
		switch ifa := ifa.(type) {
		case *IPNet:
			if ifa == nil || ifa.IP == nil || ifa.IP.IsUnspecified() || ifa.IP.IsMulticast() || ifa.Mask == nil {
				t.Errorf("unexpected value: %#v", ifa)
				continue
			}
			prefixLen, maxPrefixLen := ifa.Mask.Size()
			if ifa.IP.To4() != nil {
				if 0 >= prefixLen || prefixLen > 8*IPv4len || maxPrefixLen != 8*IPv4len {
					t.Errorf("unexpected prefix length: %v/%v", prefixLen, maxPrefixLen)
					continue
				}
				naf4++
			} else if ifa.IP.To16() != nil {
				if 0 >= prefixLen || prefixLen > 8*IPv6len || maxPrefixLen != 8*IPv6len {
					t.Errorf("unexpected prefix length: %v/%v", prefixLen, maxPrefixLen)
					continue
				}
				naf6++
			}
			t.Logf("interface address %q", ifa.String())
		default:
			t.Errorf("unexpected type: %T", ifa)
		}
	}
	return
}

func testMulticastAddrs(t *testing.T, ifmat []Addr) (nmaf4, nmaf6 int) {
	for _, ifma := range ifmat {
		switch ifma := ifma.(type) {
		case *IPAddr:
			if ifma == nil || ifma.IP == nil || ifma.IP.IsUnspecified() || !ifma.IP.IsMulticast() {
				t.Errorf("unexpected value: %#v", ifma)
				continue
			}
			if ifma.IP.To4() != nil {
				nmaf4++
			} else if ifma.IP.To16() != nil {
				nmaf6++
			}
			t.Logf("joined group address %q", ifma.String())
		default:
			t.Errorf("unexpected type: %T", ifma)
		}
	}
	return
}

func BenchmarkInterfaces(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := Interfaces(); err != nil {
			b.Fatal(err)
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
			b.Fatal(err)
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
			b.Fatal(err)
		}
	}
}

func BenchmarkInterfaceAddrs(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if _, err := InterfaceAddrs(); err != nil {
			b.Fatal(err)
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
			b.Fatal(err)
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
			b.Fatal(err)
		}
	}
}

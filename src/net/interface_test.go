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
		if ifa, ok := ifa.(*IPNet); ok {
			if ifa.IP.To4() == nil && ifa.IP.IsLinkLocalUnicast() {
				return ifa.IP.String()
			}
		}
	}
	return ""
}

type routeStats struct {
	loop  int // # of active loopback interfaces
	other int // # of active other interfaces

	uni4, uni6     int // # of active connected unicast, anycast routes
	multi4, multi6 int // # of active connected multicast route clones
}

func TestInterfaces(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	var stats routeStats
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
		if ifi.Flags&FlagUp != 0 {
			if ifi.Flags&FlagLoopback != 0 {
				stats.loop++
			} else {
				stats.other++
			}
		}
		n4, n6 := testInterfaceAddrs(t, &ifi)
		stats.uni4 += n4
		stats.uni6 += n6
		n4, n6 = testInterfaceMulticastAddrs(t, &ifi)
		stats.multi4 += n4
		stats.multi6 += n6
	}
	switch runtime.GOOS {
	case "nacl", "plan9", "solaris":
	default:
		// Test the existence of connected unicast routes for
		// IPv4.
		if supportsIPv4 && stats.loop+stats.other > 0 && stats.uni4 == 0 {
			t.Errorf("num IPv4 unicast routes = 0; want >0; summary: %+v", stats)
		}
		// Test the existence of connected unicast routes for
		// IPv6. We can assume the existence of ::1/128 when
		// at least one looopback interface is installed.
		if supportsIPv6 && stats.loop > 0 && stats.uni6 == 0 {
			t.Errorf("num IPv6 unicast routes = 0; want >0; summary: %+v", stats)
		}
	}
	switch runtime.GOOS {
	case "dragonfly", "nacl", "netbsd", "openbsd", "plan9", "solaris":
	default:
		// Test the existence of connected multicast route
		// clones for IPv4. Unlike IPv6, IPv4 multicast
		// capability is not a mandatory feature, and so this
		// test is disabled.
		//if supportsIPv4 && stats.loop > 0 && stats.uni4 > 1 && stats.multi4 == 0 {
		//	t.Errorf("num IPv4 multicast route clones = 0; want >0; summary: %+v", stats)
		//}
		// Test the existence of connected multicast route
		// clones for IPv6. Some platform never uses loopback
		// interface as the nexthop for multicast routing.
		// We can assume the existence of connected multicast
		// route clones when at least two connected unicast
		// routes, ::1/128 and other, are installed.
		if supportsIPv6 && stats.loop > 0 && stats.uni6 > 1 && stats.multi6 == 0 {
			t.Errorf("num IPv6 multicast route clones = 0; want >0; summary: %+v", stats)
		}
	}
}

func TestInterfaceAddrs(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	var stats routeStats
	for _, ifi := range ift {
		if ifi.Flags&FlagUp != 0 {
			if ifi.Flags&FlagLoopback != 0 {
				stats.loop++
			} else {
				stats.other++
			}
		}
	}
	ifat, err := InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	stats.uni4, stats.uni6 = testAddrs(t, ifat)
	// Test the existence of connected unicast routes for IPv4.
	if supportsIPv4 && stats.loop+stats.other > 0 && stats.uni4 == 0 {
		t.Errorf("num IPv4 unicast routes = 0; want >0; summary: %+v", stats)
	}
	// Test the existence of connected unicast routes for IPv6.
	// We can assume the existence of ::1/128 when at least one
	// looopback interface is installed.
	if supportsIPv6 && stats.loop > 0 && stats.uni6 == 0 {
		t.Errorf("num IPv6 unicast routes = 0; want >0; summary: %+v", stats)
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
			if len(ifa.IP) != IPv6len {
				t.Errorf("should be internal representation either IPv6 or IPv6 IPv4-mapped address: %#v", ifa)
				continue
			}
			prefixLen, maxPrefixLen := ifa.Mask.Size()
			if ifa.IP.To4() != nil {
				if 0 >= prefixLen || prefixLen > 8*IPv4len || maxPrefixLen != 8*IPv4len {
					t.Errorf("unexpected prefix length: %d/%d", prefixLen, maxPrefixLen)
					continue
				}
				if ifa.IP.IsLoopback() && (prefixLen != 8 && prefixLen != 8*IPv4len) { // see RFC 1122
					t.Errorf("unexpected prefix length for IPv4 loopback: %d/%d", prefixLen, maxPrefixLen)
					continue
				}
				naf4++
			}
			if ifa.IP.To16() != nil && ifa.IP.To4() == nil {
				if 0 >= prefixLen || prefixLen > 8*IPv6len || maxPrefixLen != 8*IPv6len {
					t.Errorf("unexpected prefix length: %d/%d", prefixLen, maxPrefixLen)
					continue
				}
				if ifa.IP.IsLoopback() && prefixLen != 8*IPv6len { // see RFC 4291
					t.Errorf("unexpected prefix length for IPv6 loopback: %d/%d", prefixLen, maxPrefixLen)
					continue
				}
				naf6++
			}
			t.Logf("interface address %q", ifa.String())
		case *IPAddr:
			if ifa == nil || ifa.IP == nil || ifa.IP.IsUnspecified() || ifa.IP.IsMulticast() {
				t.Errorf("unexpected value: %#v", ifa)
				continue
			}
			if len(ifa.IP) != IPv6len {
				t.Errorf("should be internal representation either IPv6 or IPv6 IPv4-mapped address: %#v", ifa)
				continue
			}
			if ifa.IP.To4() != nil {
				naf4++
			}
			if ifa.IP.To16() != nil && ifa.IP.To4() == nil {
				naf6++
			}
			t.Logf("interface address %s", ifa.String())
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
				t.Errorf("unexpected value: %+v", ifma)
				continue
			}
			if len(ifma.IP) != IPv6len {
				t.Errorf("should be internal representation either IPv6 or IPv6 IPv4-mapped address: %#v", ifma)
				continue
			}
			if ifma.IP.To4() != nil {
				nmaf4++
			}
			if ifma.IP.To16() != nil && ifma.IP.To4() == nil {
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
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		if _, err := Interfaces(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInterfaceByIndex(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

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
	testHookUninstaller.Do(uninstallTestHooks)

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
	testHookUninstaller.Do(uninstallTestHooks)

	for i := 0; i < b.N; i++ {
		if _, err := InterfaceAddrs(); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInterfacesAndAddrs(b *testing.B) {
	testHookUninstaller.Do(uninstallTestHooks)

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
	testHookUninstaller.Do(uninstallTestHooks)

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

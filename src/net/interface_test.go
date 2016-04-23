// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"reflect"
	"runtime"
	"testing"
)

// loopbackInterface returns an available logical network interface
// for loopback tests. It returns nil if no suitable interface is
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

func TestInterfaces(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
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
		t.Logf("%s: flags=%v index=%d mtu=%d hwaddr=%v", ifi.Name, ifi.Flags, ifi.Index, ifi.MTU, ifi.HardwareAddr)
	}
}

func TestInterfaceAddrs(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	ifStats := interfaceStats(ift)
	ifat, err := InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	uniStats, err := validateInterfaceUnicastAddrs(ifat)
	if err != nil {
		t.Fatal(err)
	}
	if err := checkUnicastStats(ifStats, uniStats); err != nil {
		t.Fatal(err)
	}
}

func TestInterfaceUnicastAddrs(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	ifStats := interfaceStats(ift)
	if err != nil {
		t.Fatal(err)
	}
	var uniStats routeStats
	for _, ifi := range ift {
		ifat, err := ifi.Addrs()
		if err != nil {
			t.Fatal(ifi, err)
		}
		stats, err := validateInterfaceUnicastAddrs(ifat)
		if err != nil {
			t.Fatal(ifi, err)
		}
		uniStats.ipv4 += stats.ipv4
		uniStats.ipv6 += stats.ipv6
	}
	if err := checkUnicastStats(ifStats, &uniStats); err != nil {
		t.Fatal(err)
	}
}

func TestInterfaceMulticastAddrs(t *testing.T) {
	ift, err := Interfaces()
	if err != nil {
		t.Fatal(err)
	}
	ifStats := interfaceStats(ift)
	ifat, err := InterfaceAddrs()
	if err != nil {
		t.Fatal(err)
	}
	uniStats, err := validateInterfaceUnicastAddrs(ifat)
	if err != nil {
		t.Fatal(err)
	}
	var multiStats routeStats
	for _, ifi := range ift {
		ifmat, err := ifi.MulticastAddrs()
		if err != nil {
			t.Fatal(ifi, err)
		}
		stats, err := validateInterfaceMulticastAddrs(ifmat)
		if err != nil {
			t.Fatal(ifi, err)
		}
		multiStats.ipv4 += stats.ipv4
		multiStats.ipv6 += stats.ipv6
	}
	if err := checkMulticastStats(ifStats, uniStats, &multiStats); err != nil {
		t.Fatal(err)
	}
}

type ifStats struct {
	loop  int // # of active loopback interfaces
	other int // # of active other interfaces
}

func interfaceStats(ift []Interface) *ifStats {
	var stats ifStats
	for _, ifi := range ift {
		if ifi.Flags&FlagUp != 0 {
			if ifi.Flags&FlagLoopback != 0 {
				stats.loop++
			} else {
				stats.other++
			}
		}
	}
	return &stats
}

type routeStats struct {
	ipv4, ipv6 int // # of active connected unicast, anycast or multicast routes
}

func validateInterfaceUnicastAddrs(ifat []Addr) (*routeStats, error) {
	// Note: BSD variants allow assigning any IPv4/IPv6 address
	// prefix to IP interface. For example,
	//   - 0.0.0.0/0 through 255.255.255.255/32
	//   - ::/0 through ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff/128
	// In other words, there is no tightly-coupled combination of
	// interface address prefixes and connected routes.
	stats := new(routeStats)
	for _, ifa := range ifat {
		switch ifa := ifa.(type) {
		case *IPNet:
			if ifa == nil || ifa.IP == nil || ifa.IP.IsMulticast() || ifa.Mask == nil {
				return nil, fmt.Errorf("unexpected value: %#v", ifa)
			}
			if len(ifa.IP) != IPv6len {
				return nil, fmt.Errorf("should be internal representation either IPv6 or IPv4-mapped IPv6 address: %#v", ifa)
			}
			prefixLen, maxPrefixLen := ifa.Mask.Size()
			if ifa.IP.To4() != nil {
				if 0 >= prefixLen || prefixLen > 8*IPv4len || maxPrefixLen != 8*IPv4len {
					return nil, fmt.Errorf("unexpected prefix length: %d/%d for %#v", prefixLen, maxPrefixLen, ifa)
				}
				if ifa.IP.IsLoopback() && (prefixLen != 8 && prefixLen != 8*IPv4len) { // see RFC 1122
					return nil, fmt.Errorf("unexpected prefix length: %d/%d for %#v", prefixLen, maxPrefixLen, ifa)
				}
				stats.ipv4++
			}
			if ifa.IP.To16() != nil && ifa.IP.To4() == nil {
				if 0 >= prefixLen || prefixLen > 8*IPv6len || maxPrefixLen != 8*IPv6len {
					return nil, fmt.Errorf("unexpected prefix length: %d/%d for %#v", prefixLen, maxPrefixLen, ifa)
				}
				if ifa.IP.IsLoopback() && prefixLen != 8*IPv6len { // see RFC 4291
					return nil, fmt.Errorf("unexpected prefix length: %d/%d for %#v", prefixLen, maxPrefixLen, ifa)
				}
				stats.ipv6++
			}
		case *IPAddr:
			if ifa == nil || ifa.IP == nil || ifa.IP.IsMulticast() {
				return nil, fmt.Errorf("unexpected value: %#v", ifa)
			}
			if len(ifa.IP) != IPv6len {
				return nil, fmt.Errorf("should be internal representation either IPv6 or IPv4-mapped IPv6 address: %#v", ifa)
			}
			if ifa.IP.To4() != nil {
				stats.ipv4++
			}
			if ifa.IP.To16() != nil && ifa.IP.To4() == nil {
				stats.ipv6++
			}
		default:
			return nil, fmt.Errorf("unexpected type: %T", ifa)
		}
	}
	return stats, nil
}

func validateInterfaceMulticastAddrs(ifat []Addr) (*routeStats, error) {
	stats := new(routeStats)
	for _, ifa := range ifat {
		switch ifa := ifa.(type) {
		case *IPAddr:
			if ifa == nil || ifa.IP == nil || ifa.IP.IsUnspecified() || !ifa.IP.IsMulticast() {
				return nil, fmt.Errorf("unexpected value: %#v", ifa)
			}
			if len(ifa.IP) != IPv6len {
				return nil, fmt.Errorf("should be internal representation either IPv6 or IPv4-mapped IPv6 address: %#v", ifa)
			}
			if ifa.IP.To4() != nil {
				stats.ipv4++
			}
			if ifa.IP.To16() != nil && ifa.IP.To4() == nil {
				stats.ipv6++
			}
		default:
			return nil, fmt.Errorf("unexpected type: %T", ifa)
		}
	}
	return stats, nil
}

func checkUnicastStats(ifStats *ifStats, uniStats *routeStats) error {
	// Test the existence of connected unicast routes for IPv4.
	if supportsIPv4 && ifStats.loop+ifStats.other > 0 && uniStats.ipv4 == 0 {
		return fmt.Errorf("num IPv4 unicast routes = 0; want >0; summary: %+v, %+v", ifStats, uniStats)
	}
	// Test the existence of connected unicast routes for IPv6.
	// We can assume the existence of ::1/128 when at least one
	// loopback interface is installed.
	if supportsIPv6 && ifStats.loop > 0 && uniStats.ipv6 == 0 {
		return fmt.Errorf("num IPv6 unicast routes = 0; want >0; summary: %+v, %+v", ifStats, uniStats)
	}
	return nil
}

func checkMulticastStats(ifStats *ifStats, uniStats, multiStats *routeStats) error {
	switch runtime.GOOS {
	case "dragonfly", "nacl", "netbsd", "openbsd", "plan9", "solaris":
	default:
		// Test the existence of connected multicast route
		// clones for IPv4. Unlike IPv6, IPv4 multicast
		// capability is not a mandatory feature, and so IPv4
		// multicast validation is ignored and we only check
		// IPv6 below.
		//
		// Test the existence of connected multicast route
		// clones for IPv6. Some platform never uses loopback
		// interface as the nexthop for multicast routing.
		// We can assume the existence of connected multicast
		// route clones when at least two connected unicast
		// routes, ::1/128 and other, are installed.
		if supportsIPv6 && ifStats.loop > 0 && uniStats.ipv6 > 1 && multiStats.ipv6 == 0 {
			return fmt.Errorf("num IPv6 multicast route clones = 0; want >0; summary: %+v, %+v, %+v", ifStats, uniStats, multiStats)
		}
	}
	return nil
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

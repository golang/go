// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"os"
	"runtime"
	"testing"
)

var ipv4MulticastListenerTests = []struct {
	net   string
	gaddr *UDPAddr // see RFC 4727
}{
	{"udp", &UDPAddr{IP: IPv4(224, 0, 0, 254), Port: 12345}},

	{"udp4", &UDPAddr{IP: IPv4(224, 0, 0, 254), Port: 12345}},
}

// TestIPv4MulticastListener tests both single and double listen to a
// test listener with same address family, same group address and same
// port.
func TestIPv4MulticastListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	closer := func(cs []*UDPConn) {
		for _, c := range cs {
			if c != nil {
				c.Close()
			}
		}
	}

	for _, ifi := range []*Interface{loopbackInterface(), nil} {
		// Note that multicast interface assignment by system
		// is not recommended because it usually relies on
		// routing stuff for finding out an appropriate
		// nexthop containing both network and link layer
		// adjacencies.
		if ifi == nil && !*testExternal {
			continue
		}
		for _, tt := range ipv4MulticastListenerTests {
			var err error
			cs := make([]*UDPConn, 2)
			if cs[0], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				t.Fatalf("First ListenMulticastUDP on %v failed: %v", ifi, err)
			}
			if err := checkMulticastListener(cs[0], tt.gaddr.IP); err != nil {
				closer(cs)
				t.Fatal(err)
			}
			if cs[1], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				closer(cs)
				t.Fatalf("Second ListenMulticastUDP on %v failed: %v", ifi, err)
			}
			if err := checkMulticastListener(cs[1], tt.gaddr.IP); err != nil {
				closer(cs)
				t.Fatal(err)
			}
			closer(cs)
		}
	}
}

var ipv6MulticastListenerTests = []struct {
	net   string
	gaddr *UDPAddr // see RFC 4727
}{
	{"udp", &UDPAddr{IP: ParseIP("ff01::114"), Port: 12345}},
	{"udp", &UDPAddr{IP: ParseIP("ff02::114"), Port: 12345}},
	{"udp", &UDPAddr{IP: ParseIP("ff04::114"), Port: 12345}},
	{"udp", &UDPAddr{IP: ParseIP("ff05::114"), Port: 12345}},
	{"udp", &UDPAddr{IP: ParseIP("ff08::114"), Port: 12345}},
	{"udp", &UDPAddr{IP: ParseIP("ff0e::114"), Port: 12345}},

	{"udp6", &UDPAddr{IP: ParseIP("ff01::114"), Port: 12345}},
	{"udp6", &UDPAddr{IP: ParseIP("ff02::114"), Port: 12345}},
	{"udp6", &UDPAddr{IP: ParseIP("ff04::114"), Port: 12345}},
	{"udp6", &UDPAddr{IP: ParseIP("ff05::114"), Port: 12345}},
	{"udp6", &UDPAddr{IP: ParseIP("ff08::114"), Port: 12345}},
	{"udp6", &UDPAddr{IP: ParseIP("ff0e::114"), Port: 12345}},
}

// TestIPv6MulticastListener tests both single and double listen to a
// test listener with same address family, same group address and same
// port.
func TestIPv6MulticastListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9", "solaris":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}
	if os.Getuid() != 0 {
		t.Skip("skipping test; must be root")
	}

	closer := func(cs []*UDPConn) {
		for _, c := range cs {
			if c != nil {
				c.Close()
			}
		}
	}

	for _, ifi := range []*Interface{loopbackInterface(), nil} {
		// Note that multicast interface assignment by system
		// is not recommended because it usually relies on
		// routing stuff for finding out an appropriate
		// nexthop containing both network and link layer
		// adjacencies.
		if ifi == nil && (!*testExternal || !*testIPv6) {
			continue
		}
		for _, tt := range ipv6MulticastListenerTests {
			var err error
			cs := make([]*UDPConn, 2)
			if cs[0], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				t.Fatalf("First ListenMulticastUDP on %v failed: %v", ifi, err)
			}
			if err := checkMulticastListener(cs[0], tt.gaddr.IP); err != nil {
				closer(cs)
				t.Fatal(err)
			}
			if cs[1], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				closer(cs)
				t.Fatalf("Second ListenMulticastUDP on %v failed: %v", ifi, err)
			}
			if err := checkMulticastListener(cs[1], tt.gaddr.IP); err != nil {
				closer(cs)
				t.Fatal(err)
			}
			closer(cs)
		}
	}
}

func checkMulticastListener(c *UDPConn, ip IP) error {
	if ok, err := multicastRIBContains(ip); err != nil {
		return err
	} else if !ok {
		return fmt.Errorf("%q not found in multicast RIB", ip.String())
	}
	la := c.LocalAddr()
	if la, ok := la.(*UDPAddr); !ok || la.Port == 0 {
		return fmt.Errorf("got %v; expected a proper address with non-zero port number", la)
	}
	return nil
}

func multicastRIBContains(ip IP) (bool, error) {
	switch runtime.GOOS {
	case "dragonfly", "netbsd", "openbsd", "plan9", "solaris", "windows":
		return true, nil // not implemented yet
	case "linux":
		if runtime.GOARCH == "arm" || runtime.GOARCH == "alpha" {
			return true, nil // not implemented yet
		}
	}
	ift, err := Interfaces()
	if err != nil {
		return false, err
	}
	for _, ifi := range ift {
		ifmat, err := ifi.MulticastAddrs()
		if err != nil {
			return false, err
		}
		for _, ifma := range ifmat {
			if ifma.(*IPAddr).IP.Equal(ip) {
				return true, nil
			}
		}
	}
	return false, nil
}

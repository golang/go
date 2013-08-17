// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

package net

import (
	"runtime"
	"syscall"
	"testing"
)

var listenerTests = []struct {
	net      string
	laddr    string
	ipv6     bool // test with underlying AF_INET6 socket
	wildcard bool // test with wildcard address
}{
	{net: "tcp", laddr: "", wildcard: true},
	{net: "tcp", laddr: "0.0.0.0", wildcard: true},
	{net: "tcp", laddr: "[::ffff:0.0.0.0]", wildcard: true},
	{net: "tcp", laddr: "[::]", ipv6: true, wildcard: true},

	{net: "tcp", laddr: "127.0.0.1"},
	{net: "tcp", laddr: "[::ffff:127.0.0.1]"},
	{net: "tcp", laddr: "[::1]", ipv6: true},

	{net: "tcp4", laddr: "", wildcard: true},
	{net: "tcp4", laddr: "0.0.0.0", wildcard: true},
	{net: "tcp4", laddr: "[::ffff:0.0.0.0]", wildcard: true},

	{net: "tcp4", laddr: "127.0.0.1"},
	{net: "tcp4", laddr: "[::ffff:127.0.0.1]"},

	{net: "tcp6", laddr: "", ipv6: true, wildcard: true},
	{net: "tcp6", laddr: "[::]", ipv6: true, wildcard: true},

	{net: "tcp6", laddr: "[::1]", ipv6: true},
}

// TestTCPListener tests both single and double listen to a test
// listener with same address family, same listening address and
// same port.
func TestTCPListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	for _, tt := range listenerTests {
		if tt.wildcard && (testing.Short() || !*testExternal) {
			continue
		}
		if tt.ipv6 && !supportsIPv6 {
			continue
		}
		l1, port := usableListenPort(t, tt.net, tt.laddr)
		checkFirstListener(t, tt.net, tt.laddr+":"+port, l1)
		l2, err := Listen(tt.net, tt.laddr+":"+port)
		checkSecondListener(t, tt.net, tt.laddr+":"+port, err, l2)
		l1.Close()
	}
}

// TestUDPListener tests both single and double listen to a test
// listener with same address family, same listening address and
// same port.
func TestUDPListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	toudpnet := func(net string) string {
		switch net {
		case "tcp":
			return "udp"
		case "tcp4":
			return "udp4"
		case "tcp6":
			return "udp6"
		}
		return "<nil>"
	}

	for _, tt := range listenerTests {
		if tt.wildcard && (testing.Short() || !*testExternal) {
			continue
		}
		if tt.ipv6 && !supportsIPv6 {
			continue
		}
		tt.net = toudpnet(tt.net)
		l1, port := usableListenPacketPort(t, tt.net, tt.laddr)
		checkFirstListener(t, tt.net, tt.laddr+":"+port, l1)
		l2, err := ListenPacket(tt.net, tt.laddr+":"+port)
		checkSecondListener(t, tt.net, tt.laddr+":"+port, err, l2)
		l1.Close()
	}
}

var dualStackListenerTests = []struct {
	net1     string // first listener
	laddr1   string
	net2     string // second listener
	laddr2   string
	wildcard bool  // test with wildcard address
	xerr     error // expected error value, nil or other
}{
	// Test cases and expected results for the attemping 2nd listen on the same port
	// 1st listen                2nd listen                 darwin  freebsd  linux  openbsd
	// ------------------------------------------------------------------------------------
	// "tcp"  ""                 "tcp"  ""                    -        -       -       -
	// "tcp"  ""                 "tcp"  "0.0.0.0"             -        -       -       -
	// "tcp"  "0.0.0.0"          "tcp"  ""                    -        -       -       -
	// ------------------------------------------------------------------------------------
	// "tcp"  ""                 "tcp"  "[::]"                -        -       -       ok
	// "tcp"  "[::]"             "tcp"  ""                    -        -       -       ok
	// "tcp"  "0.0.0.0"          "tcp"  "[::]"                -        -       -       ok
	// "tcp"  "[::]"             "tcp"  "0.0.0.0"             -        -       -       ok
	// "tcp"  "[::ffff:0.0.0.0]" "tcp"  "[::]"                -        -       -       ok
	// "tcp"  "[::]"             "tcp"  "[::ffff:0.0.0.0]"    -        -       -       ok
	// ------------------------------------------------------------------------------------
	// "tcp4" ""                 "tcp6" ""                    ok       ok      ok      ok
	// "tcp6" ""                 "tcp4" ""                    ok       ok      ok      ok
	// "tcp4" "0.0.0.0"          "tcp6" "[::]"                ok       ok      ok      ok
	// "tcp6" "[::]"             "tcp4" "0.0.0.0"             ok       ok      ok      ok
	// ------------------------------------------------------------------------------------
	// "tcp"  "127.0.0.1"        "tcp"  "[::1]"               ok       ok      ok      ok
	// "tcp"  "[::1]"            "tcp"  "127.0.0.1"           ok       ok      ok      ok
	// "tcp4" "127.0.0.1"        "tcp6" "[::1]"               ok       ok      ok      ok
	// "tcp6" "[::1]"            "tcp4" "127.0.0.1"           ok       ok      ok      ok
	//
	// Platform default configurations:
	// darwin, kernel version 11.3.0
	//	net.inet6.ip6.v6only=0 (overridable by sysctl or IPV6_V6ONLY option)
	// freebsd, kernel version 8.2
	//	net.inet6.ip6.v6only=1 (overridable by sysctl or IPV6_V6ONLY option)
	// linux, kernel version 3.0.0
	//	net.ipv6.bindv6only=0 (overridable by sysctl or IPV6_V6ONLY option)
	// openbsd, kernel version 5.0
	//	net.inet6.ip6.v6only=1 (overriding is prohibited)

	{net1: "tcp", laddr1: "", net2: "tcp", laddr2: "", wildcard: true, xerr: syscall.EADDRINUSE},
	{net1: "tcp", laddr1: "", net2: "tcp", laddr2: "0.0.0.0", wildcard: true, xerr: syscall.EADDRINUSE},
	{net1: "tcp", laddr1: "0.0.0.0", net2: "tcp", laddr2: "", wildcard: true, xerr: syscall.EADDRINUSE},

	{net1: "tcp", laddr1: "", net2: "tcp", laddr2: "[::]", wildcard: true, xerr: syscall.EADDRINUSE},
	{net1: "tcp", laddr1: "[::]", net2: "tcp", laddr2: "", wildcard: true, xerr: syscall.EADDRINUSE},
	{net1: "tcp", laddr1: "0.0.0.0", net2: "tcp", laddr2: "[::]", wildcard: true, xerr: syscall.EADDRINUSE},
	{net1: "tcp", laddr1: "[::]", net2: "tcp", laddr2: "0.0.0.0", wildcard: true, xerr: syscall.EADDRINUSE},
	{net1: "tcp", laddr1: "[::ffff:0.0.0.0]", net2: "tcp", laddr2: "[::]", wildcard: true, xerr: syscall.EADDRINUSE},
	{net1: "tcp", laddr1: "[::]", net2: "tcp", laddr2: "[::ffff:0.0.0.0]", wildcard: true, xerr: syscall.EADDRINUSE},

	{net1: "tcp4", laddr1: "", net2: "tcp6", laddr2: "", wildcard: true},
	{net1: "tcp6", laddr1: "", net2: "tcp4", laddr2: "", wildcard: true},
	{net1: "tcp4", laddr1: "0.0.0.0", net2: "tcp6", laddr2: "[::]", wildcard: true},
	{net1: "tcp6", laddr1: "[::]", net2: "tcp4", laddr2: "0.0.0.0", wildcard: true},

	{net1: "tcp", laddr1: "127.0.0.1", net2: "tcp", laddr2: "[::1]"},
	{net1: "tcp", laddr1: "[::1]", net2: "tcp", laddr2: "127.0.0.1"},
	{net1: "tcp4", laddr1: "127.0.0.1", net2: "tcp6", laddr2: "[::1]"},
	{net1: "tcp6", laddr1: "[::1]", net2: "tcp4", laddr2: "127.0.0.1"},
}

// TestDualStackTCPListener tests both single and double listen
// to a test listener with various address families, differnet
// listening address and same port.
func TestDualStackTCPListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}

	for _, tt := range dualStackListenerTests {
		if tt.wildcard && (testing.Short() || !*testExternal) {
			continue
		}
		switch runtime.GOOS {
		case "openbsd":
			if tt.wildcard && differentWildcardAddr(tt.laddr1, tt.laddr2) {
				tt.xerr = nil
			}
		}
		l1, port := usableListenPort(t, tt.net1, tt.laddr1)
		laddr := tt.laddr1 + ":" + port
		checkFirstListener(t, tt.net1, laddr, l1)
		laddr = tt.laddr2 + ":" + port
		l2, err := Listen(tt.net2, laddr)
		checkDualStackSecondListener(t, tt.net2, laddr, tt.xerr, err, l2)
		l1.Close()
	}
}

// TestDualStackUDPListener tests both single and double listen
// to a test listener with various address families, differnet
// listening address and same port.
func TestDualStackUDPListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	if !supportsIPv6 {
		t.Skip("ipv6 is not supported")
	}

	toudpnet := func(net string) string {
		switch net {
		case "tcp":
			return "udp"
		case "tcp4":
			return "udp4"
		case "tcp6":
			return "udp6"
		}
		return "<nil>"
	}

	for _, tt := range dualStackListenerTests {
		if tt.wildcard && (testing.Short() || !*testExternal) {
			continue
		}
		tt.net1 = toudpnet(tt.net1)
		tt.net2 = toudpnet(tt.net2)
		switch runtime.GOOS {
		case "openbsd":
			if tt.wildcard && differentWildcardAddr(tt.laddr1, tt.laddr2) {
				tt.xerr = nil
			}
		}
		l1, port := usableListenPacketPort(t, tt.net1, tt.laddr1)
		laddr := tt.laddr1 + ":" + port
		checkFirstListener(t, tt.net1, laddr, l1)
		laddr = tt.laddr2 + ":" + port
		l2, err := ListenPacket(tt.net2, laddr)
		checkDualStackSecondListener(t, tt.net2, laddr, tt.xerr, err, l2)
		l1.Close()
	}
}

func usableListenPort(t *testing.T, net, laddr string) (l Listener, port string) {
	var nladdr string
	var err error
	switch net {
	default:
		panic("usableListenPort net=" + net)
	case "tcp", "tcp4", "tcp6":
		l, err = Listen(net, laddr+":0")
		if err != nil {
			t.Fatalf("Probe Listen(%q, %q) failed: %v", net, laddr, err)
		}
		nladdr = l.(*TCPListener).Addr().String()
	}
	_, port, err = SplitHostPort(nladdr)
	if err != nil {
		t.Fatalf("SplitHostPort failed: %v", err)
	}
	return l, port
}

func usableListenPacketPort(t *testing.T, net, laddr string) (l PacketConn, port string) {
	var nladdr string
	var err error
	switch net {
	default:
		panic("usableListenPacketPort net=" + net)
	case "udp", "udp4", "udp6":
		l, err = ListenPacket(net, laddr+":0")
		if err != nil {
			t.Fatalf("Probe ListenPacket(%q, %q) failed: %v", net, laddr, err)
		}
		nladdr = l.(*UDPConn).LocalAddr().String()
	}
	_, port, err = SplitHostPort(nladdr)
	if err != nil {
		t.Fatalf("SplitHostPort failed: %v", err)
	}
	return l, port
}

func differentWildcardAddr(i, j string) bool {
	if (i == "" || i == "0.0.0.0" || i == "::ffff:0.0.0.0") && (j == "" || j == "0.0.0.0" || j == "::ffff:0.0.0.0") {
		return false
	}
	if i == "[::]" && j == "[::]" {
		return false
	}
	return true
}

func checkFirstListener(t *testing.T, net, laddr string, l interface{}) {
	switch net {
	case "tcp":
		fd := l.(*TCPListener).fd
		checkDualStackAddrFamily(t, net, laddr, fd)
	case "tcp4":
		fd := l.(*TCPListener).fd
		if fd.family != syscall.AF_INET {
			t.Fatalf("First Listen(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, syscall.AF_INET)
		}
	case "tcp6":
		fd := l.(*TCPListener).fd
		if fd.family != syscall.AF_INET6 {
			t.Fatalf("First Listen(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, syscall.AF_INET6)
		}
	case "udp":
		fd := l.(*UDPConn).fd
		checkDualStackAddrFamily(t, net, laddr, fd)
	case "udp4":
		fd := l.(*UDPConn).fd
		if fd.family != syscall.AF_INET {
			t.Fatalf("First ListenPacket(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, syscall.AF_INET)
		}
	case "udp6":
		fd := l.(*UDPConn).fd
		if fd.family != syscall.AF_INET6 {
			t.Fatalf("First ListenPacket(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, syscall.AF_INET6)
		}
	default:
		t.Fatalf("Unexpected network: %q", net)
	}
}

func checkSecondListener(t *testing.T, net, laddr string, err error, l interface{}) {
	switch net {
	case "tcp", "tcp4", "tcp6":
		if err == nil {
			l.(*TCPListener).Close()
			t.Fatalf("Second Listen(%q, %q) should fail", net, laddr)
		}
	case "udp", "udp4", "udp6":
		if err == nil {
			l.(*UDPConn).Close()
			t.Fatalf("Second ListenPacket(%q, %q) should fail", net, laddr)
		}
	default:
		t.Fatalf("Unexpected network: %q", net)
	}
}

func checkDualStackSecondListener(t *testing.T, net, laddr string, xerr, err error, l interface{}) {
	switch net {
	case "tcp", "tcp4", "tcp6":
		if xerr == nil && err != nil || xerr != nil && err == nil {
			t.Fatalf("Second Listen(%q, %q) returns %v, expected %v", net, laddr, err, xerr)
		}
		if err == nil {
			l.(*TCPListener).Close()
		}
	case "udp", "udp4", "udp6":
		if xerr == nil && err != nil || xerr != nil && err == nil {
			t.Fatalf("Second ListenPacket(%q, %q) returns %v, expected %v", net, laddr, err, xerr)
		}
		if err == nil {
			l.(*UDPConn).Close()
		}
	default:
		t.Fatalf("Unexpected network: %q", net)
	}
}

func checkDualStackAddrFamily(t *testing.T, net, laddr string, fd *netFD) {
	switch a := fd.laddr.(type) {
	case *TCPAddr:
		// If a node under test supports both IPv6 capability
		// and IPv6 IPv4-mapping capability, we can assume
		// that the node listens on a wildcard address with an
		// AF_INET6 socket.
		if supportsIPv4map && fd.laddr.(*TCPAddr).isWildcard() {
			if fd.family != syscall.AF_INET6 {
				t.Fatalf("Listen(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, syscall.AF_INET6)
			}
		} else {
			if fd.family != a.family() {
				t.Fatalf("Listen(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, a.family())
			}
		}
	case *UDPAddr:
		// If a node under test supports both IPv6 capability
		// and IPv6 IPv4-mapping capability, we can assume
		// that the node listens on a wildcard address with an
		// AF_INET6 socket.
		if supportsIPv4map && fd.laddr.(*UDPAddr).isWildcard() {
			if fd.family != syscall.AF_INET6 {
				t.Fatalf("ListenPacket(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, syscall.AF_INET6)
			}
		} else {
			if fd.family != a.family() {
				t.Fatalf("ListenPacket(%q, %q) returns address family %v, expected %v", net, laddr, fd.family, a.family())
			}
		}
	default:
		t.Fatalf("Unexpected protocol address type: %T", a)
	}
}

var prohibitionaryDialArgTests = []struct {
	net  string
	addr string
}{
	{"tcp6", "127.0.0.1"},
	{"tcp6", "[::ffff:127.0.0.1]"},
}

func TestProhibitionaryDialArgs(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}
	// This test requires both IPv6 and IPv6 IPv4-mapping functionality.
	if !supportsIPv4map || testing.Short() || !*testExternal {
		return
	}

	l, port := usableListenPort(t, "tcp", "[::]")
	defer l.Close()

	for _, tt := range prohibitionaryDialArgTests {
		c, err := Dial(tt.net, tt.addr+":"+port)
		if err == nil {
			c.Close()
			t.Fatalf("Dial(%q, %q) should fail", tt.net, tt.addr)
		}
	}
}

func TestWildWildcardListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("skipping test on %q", runtime.GOOS)
	}

	if testing.Short() || !*testExternal {
		t.Skip("skipping test to avoid external network")
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("Listen, ListenPacket or protocol-specific Listen panicked: %v", p)
		}
	}()

	if ln, err := Listen("tcp", ""); err == nil {
		ln.Close()
	}
	if ln, err := ListenPacket("udp", ""); err == nil {
		ln.Close()
	}
	if ln, err := ListenTCP("tcp", nil); err == nil {
		ln.Close()
	}
	if ln, err := ListenUDP("udp", nil); err == nil {
		ln.Close()
	}
	if ln, err := ListenIP("ip:icmp", nil); err == nil {
		ln.Close()
	}
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !plan9

package net

import (
	"fmt"
	"internal/testenv"
	"os"
	"runtime"
	"syscall"
	"testing"
)

func (ln *TCPListener) port() string {
	_, port, err := SplitHostPort(ln.Addr().String())
	if err != nil {
		return ""
	}
	return port
}

func (c *UDPConn) port() string {
	_, port, err := SplitHostPort(c.LocalAddr().String())
	if err != nil {
		return ""
	}
	return port
}

var tcpListenerTests = []struct {
	network string
	address string
}{
	{"tcp", ""},
	{"tcp", "0.0.0.0"},
	{"tcp", "::ffff:0.0.0.0"},
	{"tcp", "::"},

	{"tcp", "127.0.0.1"},
	{"tcp", "::ffff:127.0.0.1"},
	{"tcp", "::1"},

	{"tcp4", ""},
	{"tcp4", "0.0.0.0"},
	{"tcp4", "::ffff:0.0.0.0"},

	{"tcp4", "127.0.0.1"},
	{"tcp4", "::ffff:127.0.0.1"},

	{"tcp6", ""},
	{"tcp6", "::"},

	{"tcp6", "::1"},
}

// TestTCPListener tests both single and double listen to a test
// listener with same address family, same listening address and
// same port.
func TestTCPListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, tt := range tcpListenerTests {
		if !testableListenArgs(tt.network, JoinHostPort(tt.address, "0"), "") {
			t.Logf("skipping %s test", tt.network+" "+tt.address)
			continue
		}

		ln1, err := Listen(tt.network, JoinHostPort(tt.address, "0"))
		if err != nil {
			t.Fatal(err)
		}
		if err := checkFirstListener(tt.network, ln1); err != nil {
			ln1.Close()
			t.Fatal(err)
		}
		ln2, err := Listen(tt.network, JoinHostPort(tt.address, ln1.(*TCPListener).port()))
		if err == nil {
			ln2.Close()
		}
		if err := checkSecondListener(tt.network, tt.address, err); err != nil {
			ln1.Close()
			t.Fatal(err)
		}
		ln1.Close()
	}
}

var udpListenerTests = []struct {
	network string
	address string
}{
	{"udp", ""},
	{"udp", "0.0.0.0"},
	{"udp", "::ffff:0.0.0.0"},
	{"udp", "::"},

	{"udp", "127.0.0.1"},
	{"udp", "::ffff:127.0.0.1"},
	{"udp", "::1"},

	{"udp4", ""},
	{"udp4", "0.0.0.0"},
	{"udp4", "::ffff:0.0.0.0"},

	{"udp4", "127.0.0.1"},
	{"udp4", "::ffff:127.0.0.1"},

	{"udp6", ""},
	{"udp6", "::"},

	{"udp6", "::1"},
}

// TestUDPListener tests both single and double listen to a test
// listener with same address family, same listening address and
// same port.
func TestUDPListener(t *testing.T) {
	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	for _, tt := range udpListenerTests {
		if !testableListenArgs(tt.network, JoinHostPort(tt.address, "0"), "") {
			t.Logf("skipping %s test", tt.network+" "+tt.address)
			continue
		}

		c1, err := ListenPacket(tt.network, JoinHostPort(tt.address, "0"))
		if err != nil {
			t.Fatal(err)
		}
		if err := checkFirstListener(tt.network, c1); err != nil {
			c1.Close()
			t.Fatal(err)
		}
		c2, err := ListenPacket(tt.network, JoinHostPort(tt.address, c1.(*UDPConn).port()))
		if err == nil {
			c2.Close()
		}
		if err := checkSecondListener(tt.network, tt.address, err); err != nil {
			c1.Close()
			t.Fatal(err)
		}
		c1.Close()
	}
}

var dualStackTCPListenerTests = []struct {
	network1, address1 string // first listener
	network2, address2 string // second listener
	xerr               error  // expected error value, nil or other
}{
	// Test cases and expected results for the attempting 2nd listen on the same port
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

	{"tcp", "", "tcp", "", syscall.EADDRINUSE},
	{"tcp", "", "tcp", "0.0.0.0", syscall.EADDRINUSE},
	{"tcp", "0.0.0.0", "tcp", "", syscall.EADDRINUSE},

	{"tcp", "", "tcp", "::", syscall.EADDRINUSE},
	{"tcp", "::", "tcp", "", syscall.EADDRINUSE},
	{"tcp", "0.0.0.0", "tcp", "::", syscall.EADDRINUSE},
	{"tcp", "::", "tcp", "0.0.0.0", syscall.EADDRINUSE},
	{"tcp", "::ffff:0.0.0.0", "tcp", "::", syscall.EADDRINUSE},
	{"tcp", "::", "tcp", "::ffff:0.0.0.0", syscall.EADDRINUSE},

	{"tcp4", "", "tcp6", "", nil},
	{"tcp6", "", "tcp4", "", nil},
	{"tcp4", "0.0.0.0", "tcp6", "::", nil},
	{"tcp6", "::", "tcp4", "0.0.0.0", nil},

	{"tcp", "127.0.0.1", "tcp", "::1", nil},
	{"tcp", "::1", "tcp", "127.0.0.1", nil},
	{"tcp4", "127.0.0.1", "tcp6", "::1", nil},
	{"tcp6", "::1", "tcp4", "127.0.0.1", nil},
}

// TestDualStackTCPListener tests both single and double listen
// to a test listener with various address families, different
// listening address and same port.
//
// On DragonFly BSD, we expect the kernel version of node under test
// to be greater than or equal to 4.4.
func TestDualStackTCPListener(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv4 || !supportsIPv6 {
		t.Skip("both IPv4 and IPv6 are required")
	}

	for _, tt := range dualStackTCPListenerTests {
		if !testableListenArgs(tt.network1, JoinHostPort(tt.address1, "0"), "") {
			t.Logf("skipping %s test", tt.network1+" "+tt.address1)
			continue
		}

		if !supportsIPv4map && differentWildcardAddr(tt.address1, tt.address2) {
			tt.xerr = nil
		}
		var firstErr, secondErr error
		for i := 0; i < 5; i++ {
			lns, err := newDualStackListener()
			if err != nil {
				t.Fatal(err)
			}
			port := lns[0].port()
			for _, ln := range lns {
				ln.Close()
			}
			var ln1 Listener
			ln1, firstErr = Listen(tt.network1, JoinHostPort(tt.address1, port))
			if firstErr != nil {
				continue
			}
			if err := checkFirstListener(tt.network1, ln1); err != nil {
				ln1.Close()
				t.Fatal(err)
			}
			ln2, err := Listen(tt.network2, JoinHostPort(tt.address2, ln1.(*TCPListener).port()))
			if err == nil {
				ln2.Close()
			}
			if secondErr = checkDualStackSecondListener(tt.network2, tt.address2, err, tt.xerr); secondErr != nil {
				ln1.Close()
				continue
			}
			ln1.Close()
			break
		}
		if firstErr != nil {
			t.Error(firstErr)
		}
		if secondErr != nil {
			t.Error(secondErr)
		}
	}
}

var dualStackUDPListenerTests = []struct {
	network1, address1 string // first listener
	network2, address2 string // second listener
	xerr               error  // expected error value, nil or other
}{
	{"udp", "", "udp", "", syscall.EADDRINUSE},
	{"udp", "", "udp", "0.0.0.0", syscall.EADDRINUSE},
	{"udp", "0.0.0.0", "udp", "", syscall.EADDRINUSE},

	{"udp", "", "udp", "::", syscall.EADDRINUSE},
	{"udp", "::", "udp", "", syscall.EADDRINUSE},
	{"udp", "0.0.0.0", "udp", "::", syscall.EADDRINUSE},
	{"udp", "::", "udp", "0.0.0.0", syscall.EADDRINUSE},
	{"udp", "::ffff:0.0.0.0", "udp", "::", syscall.EADDRINUSE},
	{"udp", "::", "udp", "::ffff:0.0.0.0", syscall.EADDRINUSE},

	{"udp4", "", "udp6", "", nil},
	{"udp6", "", "udp4", "", nil},
	{"udp4", "0.0.0.0", "udp6", "::", nil},
	{"udp6", "::", "udp4", "0.0.0.0", nil},

	{"udp", "127.0.0.1", "udp", "::1", nil},
	{"udp", "::1", "udp", "127.0.0.1", nil},
	{"udp4", "127.0.0.1", "udp6", "::1", nil},
	{"udp6", "::1", "udp4", "127.0.0.1", nil},
}

// TestDualStackUDPListener tests both single and double listen
// to a test listener with various address families, different
// listening address and same port.
//
// On DragonFly BSD, we expect the kernel version of node under test
// to be greater than or equal to 4.4.
func TestDualStackUDPListener(t *testing.T) {
	switch runtime.GOOS {
	case "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}
	if !supportsIPv4 || !supportsIPv6 {
		t.Skip("both IPv4 and IPv6 are required")
	}

	for _, tt := range dualStackUDPListenerTests {
		if !testableListenArgs(tt.network1, JoinHostPort(tt.address1, "0"), "") {
			t.Logf("skipping %s test", tt.network1+" "+tt.address1)
			continue
		}

		if !supportsIPv4map && differentWildcardAddr(tt.address1, tt.address2) {
			tt.xerr = nil
		}
		var firstErr, secondErr error
		for i := 0; i < 5; i++ {
			cs, err := newDualStackPacketListener()
			if err != nil {
				t.Fatal(err)
			}
			port := cs[0].port()
			for _, c := range cs {
				c.Close()
			}
			var c1 PacketConn
			c1, firstErr = ListenPacket(tt.network1, JoinHostPort(tt.address1, port))
			if firstErr != nil {
				continue
			}
			if err := checkFirstListener(tt.network1, c1); err != nil {
				c1.Close()
				t.Fatal(err)
			}
			c2, err := ListenPacket(tt.network2, JoinHostPort(tt.address2, c1.(*UDPConn).port()))
			if err == nil {
				c2.Close()
			}
			if secondErr = checkDualStackSecondListener(tt.network2, tt.address2, err, tt.xerr); secondErr != nil {
				c1.Close()
				continue
			}
			c1.Close()
			break
		}
		if firstErr != nil {
			t.Error(firstErr)
		}
		if secondErr != nil {
			t.Error(secondErr)
		}
	}
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

func checkFirstListener(network string, ln interface{}) error {
	switch network {
	case "tcp":
		fd := ln.(*TCPListener).fd
		if err := checkDualStackAddrFamily(fd); err != nil {
			return err
		}
	case "tcp4":
		fd := ln.(*TCPListener).fd
		if fd.family != syscall.AF_INET {
			return fmt.Errorf("%v got %v; want %v", fd.laddr, fd.family, syscall.AF_INET)
		}
	case "tcp6":
		fd := ln.(*TCPListener).fd
		if fd.family != syscall.AF_INET6 {
			return fmt.Errorf("%v got %v; want %v", fd.laddr, fd.family, syscall.AF_INET6)
		}
	case "udp":
		fd := ln.(*UDPConn).fd
		if err := checkDualStackAddrFamily(fd); err != nil {
			return err
		}
	case "udp4":
		fd := ln.(*UDPConn).fd
		if fd.family != syscall.AF_INET {
			return fmt.Errorf("%v got %v; want %v", fd.laddr, fd.family, syscall.AF_INET)
		}
	case "udp6":
		fd := ln.(*UDPConn).fd
		if fd.family != syscall.AF_INET6 {
			return fmt.Errorf("%v got %v; want %v", fd.laddr, fd.family, syscall.AF_INET6)
		}
	default:
		return UnknownNetworkError(network)
	}
	return nil
}

func checkSecondListener(network, address string, err error) error {
	switch network {
	case "tcp", "tcp4", "tcp6":
		if err == nil {
			return fmt.Errorf("%s should fail", network+" "+address)
		}
	case "udp", "udp4", "udp6":
		if err == nil {
			return fmt.Errorf("%s should fail", network+" "+address)
		}
	default:
		return UnknownNetworkError(network)
	}
	return nil
}

func checkDualStackSecondListener(network, address string, err, xerr error) error {
	switch network {
	case "tcp", "tcp4", "tcp6":
		if xerr == nil && err != nil || xerr != nil && err == nil {
			return fmt.Errorf("%s got %v; want %v", network+" "+address, err, xerr)
		}
	case "udp", "udp4", "udp6":
		if xerr == nil && err != nil || xerr != nil && err == nil {
			return fmt.Errorf("%s got %v; want %v", network+" "+address, err, xerr)
		}
	default:
		return UnknownNetworkError(network)
	}
	return nil
}

func checkDualStackAddrFamily(fd *netFD) error {
	switch a := fd.laddr.(type) {
	case *TCPAddr:
		// If a node under test supports both IPv6 capability
		// and IPv6 IPv4-mapping capability, we can assume
		// that the node listens on a wildcard address with an
		// AF_INET6 socket.
		if supportsIPv4map && fd.laddr.(*TCPAddr).isWildcard() {
			if fd.family != syscall.AF_INET6 {
				return fmt.Errorf("Listen(%s, %v) returns %v; want %v", fd.net, fd.laddr, fd.family, syscall.AF_INET6)
			}
		} else {
			if fd.family != a.family() {
				return fmt.Errorf("Listen(%s, %v) returns %v; want %v", fd.net, fd.laddr, fd.family, a.family())
			}
		}
	case *UDPAddr:
		// If a node under test supports both IPv6 capability
		// and IPv6 IPv4-mapping capability, we can assume
		// that the node listens on a wildcard address with an
		// AF_INET6 socket.
		if supportsIPv4map && fd.laddr.(*UDPAddr).isWildcard() {
			if fd.family != syscall.AF_INET6 {
				return fmt.Errorf("ListenPacket(%s, %v) returns %v; want %v", fd.net, fd.laddr, fd.family, syscall.AF_INET6)
			}
		} else {
			if fd.family != a.family() {
				return fmt.Errorf("ListenPacket(%s, %v) returns %v; want %v", fd.net, fd.laddr, fd.family, a.family())
			}
		}
	default:
		return fmt.Errorf("unexpected protocol address type: %T", a)
	}
	return nil
}

func TestWildWildcardListener(t *testing.T) {
	testenv.MustHaveExternalNetwork(t)

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	}

	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("panicked: %v", p)
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
	testenv.MustHaveExternalNetwork(t)

	switch runtime.GOOS {
	case "android", "nacl", "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	case "solaris":
		t.Skipf("not supported on solaris, see golang.org/issue/7399")
	}
	if !supportsIPv4 {
		t.Skip("IPv4 is not supported")
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
		if ifi == nil || !*testIPv4 {
			continue
		}
		for _, tt := range ipv4MulticastListenerTests {
			var err error
			cs := make([]*UDPConn, 2)
			if cs[0], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				t.Fatal(err)
			}
			if err := checkMulticastListener(cs[0], tt.gaddr.IP); err != nil {
				closer(cs)
				t.Fatal(err)
			}
			if cs[1], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				closer(cs)
				t.Fatal(err)
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
	testenv.MustHaveExternalNetwork(t)

	switch runtime.GOOS {
	case "plan9":
		t.Skipf("not supported on %s", runtime.GOOS)
	case "solaris":
		t.Skipf("not supported on solaris, see issue 7399")
	}
	if !supportsIPv6 {
		t.Skip("IPv6 is not supported")
	}
	if os.Getuid() != 0 {
		t.Skip("must be root")
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
		if ifi == nil && !*testIPv6 {
			continue
		}
		for _, tt := range ipv6MulticastListenerTests {
			var err error
			cs := make([]*UDPConn, 2)
			if cs[0], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				t.Fatal(err)
			}
			if err := checkMulticastListener(cs[0], tt.gaddr.IP); err != nil {
				closer(cs)
				t.Fatal(err)
			}
			if cs[1], err = ListenMulticastUDP(tt.net, ifi, tt.gaddr); err != nil {
				closer(cs)
				t.Fatal(err)
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
		return fmt.Errorf("%s not found in multicast rib", ip.String())
	}
	la := c.LocalAddr()
	if la, ok := la.(*UDPAddr); !ok || la.Port == 0 {
		return fmt.Errorf("got %v; want a proper address with non-zero port number", la)
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

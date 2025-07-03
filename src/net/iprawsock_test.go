// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"internal/testenv"
	"reflect"
	"testing"
)

// The full stack test cases for IPConn have been moved to the
// following:
//	golang.org/x/net/ipv4
//	golang.org/x/net/ipv6
//	golang.org/x/net/icmp

type resolveIPAddrTest struct {
	network       string
	litAddrOrName string
	addr          *IPAddr
	err           error
}

var resolveIPAddrTests = []resolveIPAddrTest{
	{"ip", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
	{"ip4", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},
	{"ip4:icmp", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil},

	{"ip", "::1", &IPAddr{IP: ParseIP("::1")}, nil},
	{"ip6", "::1", &IPAddr{IP: ParseIP("::1")}, nil},
	{"ip6:ipv6-icmp", "::1", &IPAddr{IP: ParseIP("::1")}, nil},
	{"ip6:IPv6-ICMP", "::1", &IPAddr{IP: ParseIP("::1")}, nil},

	{"ip", "::1%en0", &IPAddr{IP: ParseIP("::1"), Zone: "en0"}, nil},
	{"ip6", "::1%911", &IPAddr{IP: ParseIP("::1"), Zone: "911"}, nil},

	{"", "127.0.0.1", &IPAddr{IP: IPv4(127, 0, 0, 1)}, nil}, // Go 1.0 behavior
	{"", "::1", &IPAddr{IP: ParseIP("::1")}, nil},           // Go 1.0 behavior

	{"ip4:icmp", "", &IPAddr{}, nil},

	{"l2tp", "127.0.0.1", nil, UnknownNetworkError("l2tp")},
	{"l2tp:gre", "127.0.0.1", nil, UnknownNetworkError("l2tp:gre")},
	{"tcp", "1.2.3.4:123", nil, UnknownNetworkError("tcp")},

	{"ip4", "2001:db8::1", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "2001:db8::1"}},
	{"ip4:icmp", "2001:db8::1", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "2001:db8::1"}},
	{"ip6", "127.0.0.1", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "127.0.0.1"}},
	{"ip6", "::ffff:127.0.0.1", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "::ffff:127.0.0.1"}},
	{"ip6:ipv6-icmp", "127.0.0.1", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "127.0.0.1"}},
	{"ip6:ipv6-icmp", "::ffff:127.0.0.1", nil, &AddrError{Err: errNoSuitableAddress.Error(), Addr: "::ffff:127.0.0.1"}},
}

func TestResolveIPAddr(t *testing.T) {
	if !testableNetwork("ip+nopriv") {
		t.Skip("ip+nopriv test")
	}

	origTestHookLookupIP := testHookLookupIP
	defer func() { testHookLookupIP = origTestHookLookupIP }()
	testHookLookupIP = lookupLocalhost

	for _, tt := range resolveIPAddrTests {
		addr, err := ResolveIPAddr(tt.network, tt.litAddrOrName)
		if !reflect.DeepEqual(addr, tt.addr) || !reflect.DeepEqual(err, tt.err) {
			t.Errorf("ResolveIPAddr(%q, %q) = %#v, %v, want %#v, %v", tt.network, tt.litAddrOrName, addr, err, tt.addr, tt.err)
			continue
		}
		if err == nil {
			addr2, err := ResolveIPAddr(addr.Network(), addr.String())
			if !reflect.DeepEqual(addr2, tt.addr) || err != tt.err {
				t.Errorf("(%q, %q): ResolveIPAddr(%q, %q) = %#v, %v, want %#v, %v", tt.network, tt.litAddrOrName, addr.Network(), addr.String(), addr2, err, tt.addr, tt.err)
			}
		}
	}
}

var ipConnLocalNameTests = []struct {
	net   string
	laddr *IPAddr
}{
	{"ip4:icmp", &IPAddr{IP: IPv4(127, 0, 0, 1)}},
	{"ip4:icmp", &IPAddr{}},
	{"ip4:icmp", nil},
}

func TestIPConnLocalName(t *testing.T) {
	for _, tt := range ipConnLocalNameTests {
		if !testableNetwork(tt.net) {
			t.Logf("skipping %s test", tt.net)
			continue
		}
		c, err := ListenIP(tt.net, tt.laddr)
		if testenv.SyscallIsNotSupported(err) {
			// May be inside a container that disallows creating a socket.
			t.Logf("skipping %s test: %v", tt.net, err)
			continue
		} else if err != nil {
			t.Fatal(err)
		}
		defer c.Close()
		if la := c.LocalAddr(); la == nil {
			t.Fatal("should not fail")
		}
	}
}

func TestIPConnRemoteName(t *testing.T) {
	network := "ip:tcp"
	if !testableNetwork(network) {
		t.Skipf("skipping %s test", network)
	}

	raddr := &IPAddr{IP: IPv4(127, 0, 0, 1).To4()}
	c, err := DialIP(network, &IPAddr{IP: IPv4(127, 0, 0, 1)}, raddr)
	if testenv.SyscallIsNotSupported(err) {
		// May be inside a container that disallows creating a socket.
		t.Skipf("skipping %s test: %v", network, err)
	} else if err != nil {
		t.Fatal(err)
	}
	defer c.Close()
	if !reflect.DeepEqual(raddr, c.RemoteAddr()) {
		t.Fatalf("got %#v; want %#v", c.RemoteAddr(), raddr)
	}
}

func TestDialListenIPArgs(t *testing.T) {
	type test struct {
		argLists   [][2]string
		shouldFail bool
	}
	tests := []test{
		{
			argLists: [][2]string{
				{"ip", "127.0.0.1"},
				{"ip:", "127.0.0.1"},
				{"ip::", "127.0.0.1"},
				{"ip", "::1"},
				{"ip:", "::1"},
				{"ip::", "::1"},
				{"ip4", "127.0.0.1"},
				{"ip4:", "127.0.0.1"},
				{"ip4::", "127.0.0.1"},
				{"ip6", "::1"},
				{"ip6:", "::1"},
				{"ip6::", "::1"},
			},
			shouldFail: true,
		},
	}
	if testableNetwork("ip") {
		priv := test{shouldFail: false}
		for _, tt := range []struct {
			network, address string
			args             [2]string
		}{
			{"ip4:47", "127.0.0.1", [2]string{"ip4:47", "127.0.0.1"}},
			{"ip6:47", "::1", [2]string{"ip6:47", "::1"}},
		} {
			c, err := ListenPacket(tt.network, tt.address)
			if err != nil {
				continue
			}
			c.Close()
			priv.argLists = append(priv.argLists, tt.args)
		}
		if len(priv.argLists) > 0 {
			tests = append(tests, priv)
		}
	}

	for _, tt := range tests {
		for _, args := range tt.argLists {
			_, err := Dial(args[0], args[1])
			if tt.shouldFail != (err != nil) {
				t.Errorf("Dial(%q, %q) = %v; want (err != nil) is %t", args[0], args[1], err, tt.shouldFail)
			}
			_, err = ListenPacket(args[0], args[1])
			if tt.shouldFail != (err != nil) {
				t.Errorf("ListenPacket(%q, %q) = %v; want (err != nil) is %t", args[0], args[1], err, tt.shouldFail)
			}
			a, err := ResolveIPAddr("ip", args[1])
			if err != nil {
				t.Errorf("ResolveIPAddr(\"ip\", %q) = %v", args[1], err)
				continue
			}
			_, err = DialIP(args[0], nil, a)
			if tt.shouldFail != (err != nil) {
				t.Errorf("DialIP(%q, %v) = %v; want (err != nil) is %t", args[0], a, err, tt.shouldFail)
			}
			_, err = ListenIP(args[0], a)
			if tt.shouldFail != (err != nil) {
				t.Errorf("ListenIP(%q, %v) = %v; want (err != nil) is %t", args[0], a, err, tt.shouldFail)
			}
		}
	}
}

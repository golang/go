// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"fmt"
	"io"
	"strings"
	"testing"
)

func TestResolveGoogle(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !supportsIPv6 || !*testIPv4 || !*testIPv6 {
		t.Skip("both IPv4 and IPv6 are required")
	}

	for _, network := range []string{"tcp", "tcp4", "tcp6"} {
		addr, err := ResolveTCPAddr(network, "www.google.com:http")
		if err != nil {
			t.Error(err)
			continue
		}
		switch {
		case network == "tcp" && addr.IP.To4() == nil:
			fallthrough
		case network == "tcp4" && addr.IP.To4() == nil:
			t.Errorf("got %v; want an IPv4 address on %s", addr, network)
		case network == "tcp6" && (addr.IP.To16() == nil || addr.IP.To4() != nil):
			t.Errorf("got %v; want an IPv6 address on %s", addr, network)
		}
	}
}

var dialGoogleTests = []struct {
	dial               func(string, string) (Conn, error)
	unreachableNetwork string
	networks           []string
	addrs              []string
}{
	{
		dial:     (&Dialer{DualStack: true}).Dial,
		networks: []string{"tcp", "tcp4", "tcp6"},
		addrs:    []string{"www.google.com:http"},
	},
	{
		dial:               Dial,
		unreachableNetwork: "tcp6",
		networks:           []string{"tcp", "tcp4"},
	},
	{
		dial:               Dial,
		unreachableNetwork: "tcp4",
		networks:           []string{"tcp", "tcp6"},
	},
}

func TestDialGoogle(t *testing.T) {
	if testing.Short() || !*testExternal {
		t.Skip("avoid external network")
	}
	if !supportsIPv4 || !supportsIPv6 || !*testIPv4 || !*testIPv6 {
		t.Skip("both IPv4 and IPv6 are required")
	}

	var err error
	dialGoogleTests[1].addrs, dialGoogleTests[2].addrs, err = googleLiteralAddrs()
	if err != nil {
		t.Error(err)
	}
	for _, tt := range dialGoogleTests {
		for _, network := range tt.networks {
			disableSocketConnect(tt.unreachableNetwork)
			for _, addr := range tt.addrs {
				if err := fetchGoogle(tt.dial, network, addr); err != nil {
					t.Error(err)
				}
			}
			enableSocketConnect()
		}
	}
}

var (
	literalAddrs4 = [...]string{
		"%d.%d.%d.%d:80",
		"www.google.com:80",
		"%d.%d.%d.%d:http",
		"www.google.com:http",
		"%03d.%03d.%03d.%03d:0080",
		"[::ffff:%d.%d.%d.%d]:80",
		"[::ffff:%02x%02x:%02x%02x]:80",
		"[0:0:0:0:0000:ffff:%d.%d.%d.%d]:80",
		"[0:0:0:0:000000:ffff:%d.%d.%d.%d]:80",
		"[0:0:0:0::ffff:%d.%d.%d.%d]:80",
	}
	literalAddrs6 = [...]string{
		"[%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x]:80",
		"ipv6.google.com:80",
		"[%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x]:http",
		"ipv6.google.com:http",
	}
)

func googleLiteralAddrs() (lits4, lits6 []string, err error) {
	ips, err := LookupIP("www.google.com")
	if err != nil {
		return nil, nil, err
	}
	if len(ips) == 0 {
		return nil, nil, nil
	}
	var ip4, ip6 IP
	for _, ip := range ips {
		if ip4 == nil && ip.To4() != nil {
			ip4 = ip.To4()
		}
		if ip6 == nil && ip.To16() != nil && ip.To4() == nil {
			ip6 = ip.To16()
		}
		if ip4 != nil && ip6 != nil {
			break
		}
	}
	if ip4 != nil {
		for i, lit4 := range literalAddrs4 {
			if strings.Contains(lit4, "%") {
				literalAddrs4[i] = fmt.Sprintf(lit4, ip4[0], ip4[1], ip4[2], ip4[3])
			}
		}
		lits4 = literalAddrs4[:]
	}
	if ip6 != nil {
		for i, lit6 := range literalAddrs6 {
			if strings.Contains(lit6, "%") {
				literalAddrs6[i] = fmt.Sprintf(lit6, ip6[0], ip6[1], ip6[2], ip6[3], ip6[4], ip6[5], ip6[6], ip6[7], ip6[8], ip6[9], ip6[10], ip6[11], ip6[12], ip6[13], ip6[14], ip6[15])
			}
		}
		lits6 = literalAddrs6[:]
	}
	return
}

func fetchGoogle(dial func(string, string) (Conn, error), network, address string) error {
	c, err := dial(network, address)
	if err != nil {
		return err
	}
	defer c.Close()
	req := []byte("GET /robots.txt HTTP/1.0\r\nHost: www.google.com\r\n\r\n")
	if _, err := c.Write(req); err != nil {
		return err
	}
	b := make([]byte, 1000)
	n, err := io.ReadFull(c, b)
	if err != nil {
		return err
	}
	if n < 1000 {
		return fmt.Errorf("short read from %s:%s->%s", network, c.RemoteAddr(), c.LocalAddr())
	}
	return nil
}

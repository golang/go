// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(netipv6zoneTests, netipv6zone)
}

var netipv6zoneTests = []testCase{
	{
		Name: "netipv6zone.0",
		In: `package main

import "net"

var a = []struct {
	*net.IPNet
}{
	&net.IPNet{net.ParseIP("2001:DB8::"), net.IPMask(net.ParseIP("ffff:ffff:ffff::"))},
}

func f() net.Addr {
	b := net.IPNet{net.IPv4(127, 0, 0, 1), net.IPv4Mask(255, 0, 0, 0)}
	c := &net.IPAddr{ip1}
	sub(&net.UDPAddr{ip2, 12345})
	d := &net.TCPAddr{IP: ip3, Port: 54321}
	return &net.TCPAddr{ip4}, nil
}
`,
		Out: `package main

import "net"

var a = []struct {
	*net.IPNet
}{
	&net.IPNet{IP: net.ParseIP("2001:DB8::"), Mask: net.IPMask(net.ParseIP("ffff:ffff:ffff::"))},
}

func f() net.Addr {
	b := net.IPNet{IP: net.IPv4(127, 0, 0, 1), Mask: net.IPv4Mask(255, 0, 0, 0)}
	c := &net.IPAddr{IP: ip1}
	sub(&net.UDPAddr{IP: ip2, Port: 12345})
	d := &net.TCPAddr{IP: ip3, Port: 54321}
	return &net.TCPAddr{IP: ip4}, nil
}
`,
	},
}

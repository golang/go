// Copyright 2012 The Go Authors. All rights reserved.
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

func f() net.Addr {
	a := &net.IPAddr{ip1}
	sub(&net.UDPAddr{ip2, 12345})
	c := &net.TCPAddr{IP: ip3, Port: 54321}
	d := &net.TCPAddr{ip4, 0}
	p := 1234
	e := &net.TCPAddr{ip4, p}
	return &net.TCPAddr{ip5}, nil
}
`,
		Out: `package main

import "net"

func f() net.Addr {
	a := &net.IPAddr{IP: ip1}
	sub(&net.UDPAddr{IP: ip2, Port: 12345})
	c := &net.TCPAddr{IP: ip3, Port: 54321}
	d := &net.TCPAddr{IP: ip4}
	p := 1234
	e := &net.TCPAddr{IP: ip4, Port: p}
	return &net.TCPAddr{IP: ip5}, nil
}
`,
	},
}

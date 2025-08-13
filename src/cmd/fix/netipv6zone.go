// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	register(netipv6zoneFix)
}

var netipv6zoneFix = fix{
	name: "netipv6zone",
	date: "2012-11-26",
	f:    noop,
	desc: `Adapt element key to IPAddr, UDPAddr or TCPAddr composite literals (removed).

https://codereview.appspot.com/6849045/
`,
}

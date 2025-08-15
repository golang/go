// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"os"
	"sync/atomic"
	"time"
	_ "unsafe"
)

// defaultNS is the default name servers to use in the absence of DNS configuration.
//
// defaultNS should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/pojntfx/hydrapp/hydrapp
//   - github.com/mtibben/androiddnsfix
//   - github.com/metacubex/mihomo
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname defaultNS
var defaultNS = []string{"127.0.0.1:53", "[::1]:53"}

var getHostname = os.Hostname // variable for testing

type dnsConfig struct {
	servers       []string      // server addresses (in host:port form) to use
	search        []string      // rooted suffixes to append to local name
	ndots         int           // number of dots in name to trigger absolute lookup
	timeout       time.Duration // wait before giving up on a query, including retries
	attempts      int           // lost packets before giving up on server
	rotate        bool          // round robin among servers
	unknownOpt    bool          // anything unknown was encountered
	lookup        []string      // OpenBSD top-level database "lookup" order
	err           error         // any error that occurs during open of resolv.conf
	mtime         time.Time     // time of resolv.conf modification
	soffset       atomic.Uint32 // used by serverOffset
	singleRequest bool          // use sequential A and AAAA queries instead of parallel queries
	useTCP        bool          // force usage of TCP for DNS resolutions
	trustAD       bool          // add AD flag to queries
	noReload      bool          // do not check for config file updates
}

// serverOffset returns an offset that can be used to determine
// indices of servers in c.servers when making queries.
// When the rotate option is enabled, this offset increases.
// Otherwise it is always 0.
func (c *dnsConfig) serverOffset() uint32 {
	if c.rotate {
		return c.soffset.Add(1) - 1 // return 0 to start
	}
	return 0
}

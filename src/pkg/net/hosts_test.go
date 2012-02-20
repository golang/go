// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"sort"
	"testing"
)

type hostTest struct {
	host string
	ips  []IP
}

var hosttests = []hostTest{
	{"odin", []IP{
		IPv4(127, 0, 0, 2),
		IPv4(127, 0, 0, 3),
		ParseIP("::2"),
	}},
	{"thor", []IP{
		IPv4(127, 1, 1, 1),
	}},
	{"loki", []IP{}},
	{"ullr", []IP{
		IPv4(127, 1, 1, 2),
	}},
	{"ullrhost", []IP{
		IPv4(127, 1, 1, 2),
	}},
}

func TestLookupStaticHost(t *testing.T) {
	p := hostsPath
	hostsPath = "testdata/hosts"
	for i := 0; i < len(hosttests); i++ {
		tt := hosttests[i]
		ips := lookupStaticHost(tt.host)
		if len(ips) != len(tt.ips) {
			t.Errorf("# of hosts = %v; want %v",
				len(ips), len(tt.ips))
			return
		}
		for k, v := range ips {
			if tt.ips[k].String() != v {
				t.Errorf("lookupStaticHost(%q) = %v; want %v",
					tt.host, v, tt.ips[k])
			}
		}
	}
	hostsPath = p
}

func TestLookupHost(t *testing.T) {
	// Can't depend on this to return anything in particular,
	// but if it does return something, make sure it doesn't
	// duplicate addresses (a common bug due to the way
	// getaddrinfo works).
	addrs, _ := LookupHost("localhost")
	sort.Strings(addrs)
	for i := 0; i+1 < len(addrs); i++ {
		if addrs[i] == addrs[i+1] {
			t.Fatalf("LookupHost(\"localhost\") = %v, has duplicate addresses", addrs)
		}
	}
}

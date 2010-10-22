// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
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
	hostsPath = "hosts_testdata"
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

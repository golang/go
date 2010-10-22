// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"testing"
)

type portTest struct {
	netw string
	name string
	port int
	ok   bool
}

var porttests = []portTest{
	{"tcp", "echo", 7, true},
	{"tcp", "discard", 9, true},
	{"tcp", "systat", 11, true},
	{"tcp", "daytime", 13, true},
	{"tcp", "chargen", 19, true},
	{"tcp", "ftp-data", 20, true},
	{"tcp", "ftp", 21, true},
	{"tcp", "telnet", 23, true},
	{"tcp", "smtp", 25, true},
	{"tcp", "time", 37, true},
	{"tcp", "domain", 53, true},
	{"tcp", "gopher", 70, true},
	{"tcp", "finger", 79, true},
	{"tcp", "http", 80, true},

	{"udp", "echo", 7, true},
	{"udp", "tftp", 69, true},
	{"udp", "bootpc", 68, true},
	{"udp", "bootps", 67, true},
	{"udp", "domain", 53, true},
	{"udp", "ntp", 123, true},
	{"udp", "snmp", 161, true},
	{"udp", "syslog", 514, true},

	{"--badnet--", "zzz", 0, false},
	{"tcp", "--badport--", 0, false},
}

func TestLookupPort(t *testing.T) {
	for i := 0; i < len(porttests); i++ {
		tt := porttests[i]
		if port, err := LookupPort(tt.netw, tt.name); port != tt.port || (err == nil) != tt.ok {
			t.Errorf("LookupPort(%q, %q) = %v, %s; want %v",
				tt.netw, tt.name, port, err, tt.port)
		}
	}
}

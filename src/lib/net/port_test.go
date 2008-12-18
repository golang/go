// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"net";
	"testing";
)

type PortTest struct {
	netw string;
	name string;
	port int;
	ok bool;
}

var porttests = []PortTest {
	PortTest{ "tcp", "echo", 7, true },
	PortTest{ "tcp", "discard", 9, true },
	PortTest{ "tcp", "systat", 11, true },
	PortTest{ "tcp", "daytime", 13, true },
	PortTest{ "tcp", "chargen", 19, true },
	PortTest{ "tcp", "ftp-data", 20, true },
	PortTest{ "tcp", "ftp", 21, true },
	PortTest{ "tcp", "ssh", 22, true },
	PortTest{ "tcp", "telnet", 23, true },
	PortTest{ "tcp", "smtp", 25, true },
	PortTest{ "tcp", "time", 37, true },
	PortTest{ "tcp", "domain", 53, true },
	PortTest{ "tcp", "gopher", 70, true },
	PortTest{ "tcp", "finger", 79, true },
	PortTest{ "tcp", "http", 80, true },

	PortTest{ "udp", "echo", 7, true },
	PortTest{ "udp", "tacacs", 49, true },
	PortTest{ "udp", "tftp", 69, true },
	PortTest{ "udp", "bootpc", 68, true },
	PortTest{ "udp", "bootps", 67, true },
	PortTest{ "udp", "domain", 53, true },
	PortTest{ "udp", "ntp", 123, true },
	PortTest{ "udp", "snmp", 161, true },
	PortTest{ "udp", "syslog", 514, true },
	PortTest{ "udp", "nfs", 2049, true },

	PortTest{ "--badnet--", "zzz", 0, false },
	PortTest{ "tcp", "--badport--", 0, false },
}

export func TestLookupPort(t *testing.T) {
	for i := 0; i < len(porttests); i++ {
		tt := porttests[i];
		if port, ok := LookupPort(tt.netw, tt.name); port != tt.port || ok != tt.ok {
			t.Errorf("LookupPort(%q, %q) = %v, %v; want %v, %v",
				tt.netw, tt.name, port, ok, tt.port, tt.ok);
		}
	}
}

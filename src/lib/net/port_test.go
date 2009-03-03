// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"net";
	"testing";
)

type portTest struct {
	netw string;
	name string;
	port int;
	ok bool;
}

var porttests = []portTest {
	portTest{ "tcp", "echo", 7, true },
	portTest{ "tcp", "discard", 9, true },
	portTest{ "tcp", "systat", 11, true },
	portTest{ "tcp", "daytime", 13, true },
	portTest{ "tcp", "chargen", 19, true },
	portTest{ "tcp", "ftp-data", 20, true },
	portTest{ "tcp", "ftp", 21, true },
	portTest{ "tcp", "ssh", 22, true },
	portTest{ "tcp", "telnet", 23, true },
	portTest{ "tcp", "smtp", 25, true },
	portTest{ "tcp", "time", 37, true },
	portTest{ "tcp", "domain", 53, true },
	portTest{ "tcp", "gopher", 70, true },
	portTest{ "tcp", "finger", 79, true },
	portTest{ "tcp", "http", 80, true },

	portTest{ "udp", "echo", 7, true },
	portTest{ "udp", "tacacs", 49, true },
	portTest{ "udp", "tftp", 69, true },
	portTest{ "udp", "bootpc", 68, true },
	portTest{ "udp", "bootps", 67, true },
	portTest{ "udp", "domain", 53, true },
	portTest{ "udp", "ntp", 123, true },
	portTest{ "udp", "snmp", 161, true },
	portTest{ "udp", "syslog", 514, true },
	portTest{ "udp", "nfs", 2049, true },

	portTest{ "--badnet--", "zzz", 0, false },
	portTest{ "tcp", "--badport--", 0, false },
}

func TestLookupPort(t *testing.T) {
	for i := 0; i < len(porttests); i++ {
		tt := porttests[i];
		if port, ok := LookupPort(tt.netw, tt.name); port != tt.port || ok != tt.ok {
			t.Errorf("LookupPort(%q, %q) = %v, %v; want %v, %v",
				tt.netw, tt.name, port, ok, tt.port, tt.ok);
		}
	}
}

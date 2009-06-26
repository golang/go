// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"net";
	"os";
	"regexp";
	"testing";
)

type DialErrorTest struct {
	Net string;
	Laddr string;
	Raddr string;
	Pattern string;
}

var dialErrorTests = []DialErrorTest {
	DialErrorTest{
		"datakit", "", "mh/astro/r70",
		"dial datakit mh/astro/r70: unknown network datakit",
	},
	DialErrorTest{
		"tcp", "", "127.0.0.1:☺",
		"dial tcp 127.0.0.1:☺: unknown port tcp/☺",
	},
	DialErrorTest{
		"tcp", "", "no-such-name.google.com.:80",
		"dial tcp no-such-name.google.com.:80: lookup no-such-name.google.com.: no such host",
	},
	DialErrorTest{
		"tcp", "", "no-such-name.no-such-top-level-domain.:80",
		"dial tcp no-such-name.no-such-top-level-domain.:80: lookup no-such-name.no-such-top-level-domain.: no such host",
	},
	DialErrorTest{
		"tcp", "", "no-such-name:80",
		"dial tcp no-such-name:80: lookup no-such-name.google.com.: no such host",
	},
	DialErrorTest{
		"tcp", "", "mh/astro/r70:http",
		"dial tcp mh/astro/r70:http: lookup mh/astro/r70: invalid domain name",
	},
	DialErrorTest{
		"unix", "", "/etc/file-not-found",
		"dial unix /etc/file-not-found: no such file or directory",
	},
	DialErrorTest{
		"unix", "", "/etc/",
		"dial unix /etc/: (permission denied|socket operation on non-socket)",
	},
}

func TestDialError(t *testing.T) {
	for i, tt := range dialErrorTests {
		c, e := net.Dial(tt.Net, tt.Laddr, tt.Raddr);
		if c != nil {
			c.Close();
		}
		if e == nil {
			t.Errorf("#%d: nil error, want match for %#q", i, tt.Pattern);
			continue;
		}
		s := e.String();
		match, err := regexp.Match(tt.Pattern, s);
		if !match {
			t.Errorf("#%d: %q, want match for %#q", i, s, tt.Pattern);
		}
	}
}

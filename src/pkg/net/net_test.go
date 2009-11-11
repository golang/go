// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package net

import (
	"flag";
	"regexp";
	"testing";
)

var runErrorTest = flag.Bool("run_error_test", false, "let TestDialError check for dns errors")

type DialErrorTest struct {
	Net	string;
	Laddr	string;
	Raddr	string;
	Pattern	string;
}

var dialErrorTests = []DialErrorTest{
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
		"dial tcp no-such-name.google.com.:80: lookup no-such-name.google.com.( on .*)?: no (.*)",
	},
	DialErrorTest{
		"tcp", "", "no-such-name.no-such-top-level-domain.:80",
		"dial tcp no-such-name.no-such-top-level-domain.:80: lookup no-such-name.no-such-top-level-domain.( on .*)?: no (.*)",
	},
	DialErrorTest{
		"tcp", "", "no-such-name:80",
		`dial tcp no-such-name:80: lookup no-such-name\.(.*\.)?( on .*)?: no (.*)`,
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
		"dial unix /etc/: (permission denied|socket operation on non-socket|connection refused)",
	},
}

func TestDialError(t *testing.T) {
	if !*runErrorTest {
		t.Logf("test disabled; use --run_error_test to enable");
		return;
	}
	for i, tt := range dialErrorTests {
		c, e := Dial(tt.Net, tt.Laddr, tt.Raddr);
		if c != nil {
			c.Close()
		}
		if e == nil {
			t.Errorf("#%d: nil error, want match for %#q", i, tt.Pattern);
			continue;
		}
		s := e.String();
		match, _ := regexp.MatchString(tt.Pattern, s);
		if !match {
			t.Errorf("#%d: %q, want match for %#q", i, s, tt.Pattern)
		}
	}
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import (
	"strconv";
	"testing";
)

type QuoteTest struct {
	in string;
	out string;
}

var quotetests = []QuoteTest {
	QuoteTest{ "\a\b\f\r\n\t\v", `"\a\b\f\r\n\t\v"` },
	QuoteTest{ "\\", `"\\"` },
	QuoteTest{ "abc\xffdef", `"abc\xffdef"` },
	QuoteTest{ "\u263a", `"\u263a"` },
	QuoteTest{ "\U0010ffff", `"\U0010ffff"` },
}

export func TestQuote(t *testing.T) {
	for i := 0; i < len(quotetests); i++ {
		tt := quotetests[i];
		if out := Quote(tt.in); out != tt.out {
			t.Errorf("Quote(%s) = %s, want %s", tt.in, out, tt.out);
		}
	}
}

type CanBackquoteTest struct {
	in string;
	out bool;
}

var canbackquotetests = []CanBackquoteTest {
	CanBackquoteTest{ "`", false },
	CanBackquoteTest{ string(0), false },
	CanBackquoteTest{ string(1), false },
	CanBackquoteTest{ string(2), false },
	CanBackquoteTest{ string(3), false },
	CanBackquoteTest{ string(4), false },
	CanBackquoteTest{ string(5), false },
	CanBackquoteTest{ string(6), false },
	CanBackquoteTest{ string(7), false },
	CanBackquoteTest{ string(8), false },
	CanBackquoteTest{ string(9), false },
	CanBackquoteTest{ string(10), false },
	CanBackquoteTest{ string(11), false },
	CanBackquoteTest{ string(12), false },
	CanBackquoteTest{ string(13), false },
	CanBackquoteTest{ string(14), false },
	CanBackquoteTest{ string(15), false },
	CanBackquoteTest{ string(16), false },
	CanBackquoteTest{ string(17), false },
	CanBackquoteTest{ string(18), false },
	CanBackquoteTest{ string(19), false },
	CanBackquoteTest{ string(20), false },
	CanBackquoteTest{ string(21), false },
	CanBackquoteTest{ string(22), false },
	CanBackquoteTest{ string(23), false },
	CanBackquoteTest{ string(24), false },
	CanBackquoteTest{ string(25), false },
	CanBackquoteTest{ string(26), false },
	CanBackquoteTest{ string(27), false },
	CanBackquoteTest{ string(28), false },
	CanBackquoteTest{ string(29), false },
	CanBackquoteTest{ string(30), false },
	CanBackquoteTest{ string(31), false },
	CanBackquoteTest{ `' !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`, true },
	CanBackquoteTest{ `0123456789`, true },
	CanBackquoteTest{ `ABCDEFGHIJKLMNOPQRSTUVWXYZ`, true },
	CanBackquoteTest{ `abcdefghijklmnopqrstuvwxyz`, true },
	CanBackquoteTest{ `☺`, true },
}

export func TestCanBackquote(t *testing.T) {
	for i := 0; i < len(canbackquotetests); i++ {
		tt := canbackquotetests[i];
		if out := CanBackquote(tt.in); out != tt.out {
			t.Errorf("CanBackquote(%q) = %v, want %v", tt.in, out, tt.out);
		}
	}
}

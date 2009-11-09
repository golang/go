// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"os";
	. "strconv";
	"testing";
)

type quoteTest struct {
	in	string;
	out	string;
}

var quotetests = []quoteTest{
	quoteTest{"\a\b\f\r\n\t\v", `"\a\b\f\r\n\t\v"`},
	quoteTest{"\\", `"\\"`},
	quoteTest{"abc\xffdef", `"abc\xffdef"`},
	quoteTest{"\u263a", `"\u263a"`},
	quoteTest{"\U0010ffff", `"\U0010ffff"`},
	quoteTest{"\x04", `"\x04"`},
}

func TestQuote(t *testing.T) {
	for i := 0; i < len(quotetests); i++ {
		tt := quotetests[i];
		if out := Quote(tt.in); out != tt.out {
			t.Errorf("Quote(%s) = %s, want %s", tt.in, out, tt.out)
		}
	}
}

type canBackquoteTest struct {
	in	string;
	out	bool;
}

var canbackquotetests = []canBackquoteTest{
	canBackquoteTest{"`", false},
	canBackquoteTest{string(0), false},
	canBackquoteTest{string(1), false},
	canBackquoteTest{string(2), false},
	canBackquoteTest{string(3), false},
	canBackquoteTest{string(4), false},
	canBackquoteTest{string(5), false},
	canBackquoteTest{string(6), false},
	canBackquoteTest{string(7), false},
	canBackquoteTest{string(8), false},
	canBackquoteTest{string(9), true},	// \t
	canBackquoteTest{string(10), false},
	canBackquoteTest{string(11), false},
	canBackquoteTest{string(12), false},
	canBackquoteTest{string(13), false},
	canBackquoteTest{string(14), false},
	canBackquoteTest{string(15), false},
	canBackquoteTest{string(16), false},
	canBackquoteTest{string(17), false},
	canBackquoteTest{string(18), false},
	canBackquoteTest{string(19), false},
	canBackquoteTest{string(20), false},
	canBackquoteTest{string(21), false},
	canBackquoteTest{string(22), false},
	canBackquoteTest{string(23), false},
	canBackquoteTest{string(24), false},
	canBackquoteTest{string(25), false},
	canBackquoteTest{string(26), false},
	canBackquoteTest{string(27), false},
	canBackquoteTest{string(28), false},
	canBackquoteTest{string(29), false},
	canBackquoteTest{string(30), false},
	canBackquoteTest{string(31), false},
	canBackquoteTest{`' !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`, true},
	canBackquoteTest{`0123456789`, true},
	canBackquoteTest{`ABCDEFGHIJKLMNOPQRSTUVWXYZ`, true},
	canBackquoteTest{`abcdefghijklmnopqrstuvwxyz`, true},
	canBackquoteTest{`☺`, true},
}

func TestCanBackquote(t *testing.T) {
	for i := 0; i < len(canbackquotetests); i++ {
		tt := canbackquotetests[i];
		if out := CanBackquote(tt.in); out != tt.out {
			t.Errorf("CanBackquote(%q) = %v, want %v", tt.in, out, tt.out)
		}
	}
}

var unquotetests = []quoteTest{
	quoteTest{`""`, ""},
	quoteTest{`"a"`, "a"},
	quoteTest{`"abc"`, "abc"},
	quoteTest{`"☺"`, "☺"},
	quoteTest{`"hello world"`, "hello world"},
	quoteTest{`"\xFF"`, "\xFF"},
	quoteTest{`"\377"`, "\377"},
	quoteTest{`"\u1234"`, "\u1234"},
	quoteTest{`"\U00010111"`, "\U00010111"},
	quoteTest{`"\U0001011111"`, "\U0001011111"},
	quoteTest{`"\a\b\f\n\r\t\v\\\""`, "\a\b\f\n\r\t\v\\\""},
	quoteTest{`"'"`, "'"},

	quoteTest{`'a'`, "a"},
	quoteTest{`'☹'`, "☹"},
	quoteTest{`'\a'`, "\a"},
	quoteTest{`'\x10'`, "\x10"},
	quoteTest{`'\377'`, "\377"},
	quoteTest{`'\u1234'`, "\u1234"},
	quoteTest{`'\U00010111'`, "\U00010111"},
	quoteTest{`'\t'`, "\t"},
	quoteTest{`' '`, " "},
	quoteTest{`'\''`, "'"},
	quoteTest{`'"'`, "\""},

	quoteTest{"``", ``},
	quoteTest{"`a`", `a`},
	quoteTest{"`abc`", `abc`},
	quoteTest{"`☺`", `☺`},
	quoteTest{"`hello world`", `hello world`},
	quoteTest{"`\\xFF`", `\xFF`},
	quoteTest{"`\\377`", `\377`},
	quoteTest{"`\\`", `\`},
	quoteTest{"`	`", `	`},
	quoteTest{"` `", ` `},
}

var misquoted = []string{
	``,
	`"`,
	`"a`,
	`"'`,
	`b"`,
	`"\"`,
	`'\'`,
	`'ab'`,
	`"\x1!"`,
	`"\U12345678"`,
	`"\z"`,
	"`",
	"`xxx",
	"`\"",
	`"\'"`,
	`'\"'`,
}

func TestUnquote(t *testing.T) {
	for i := 0; i < len(unquotetests); i++ {
		tt := unquotetests[i];
		if out, err := Unquote(tt.in); err != nil && out != tt.out {
			t.Errorf("Unquote(%#q) = %q, %v want %q, nil", tt.in, out, err, tt.out)
		}
	}

	// run the quote tests too, backward
	for i := 0; i < len(quotetests); i++ {
		tt := quotetests[i];
		if in, err := Unquote(tt.out); in != tt.in {
			t.Errorf("Unquote(%#q) = %q, %v, want %q, nil", tt.out, in, err, tt.in)
		}
	}

	for i := 0; i < len(misquoted); i++ {
		s := misquoted[i];
		if out, err := Unquote(s); out != "" || err != os.EINVAL {
			t.Errorf("Unquote(%#q) = %q, %v want %q, %v", s, out, err, "", os.EINVAL)
		}
	}
}

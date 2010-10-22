// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	"os"
	. "strconv"
	"testing"
)

type quoteTest struct {
	in  string
	out string
}

var quotetests = []quoteTest{
	{"\a\b\f\r\n\t\v", `"\a\b\f\r\n\t\v"`},
	{"\\", `"\\"`},
	{"abc\xffdef", `"abc\xffdef"`},
	{"\u263a", `"\u263a"`},
	{"\U0010ffff", `"\U0010ffff"`},
	{"\x04", `"\x04"`},
}

func TestQuote(t *testing.T) {
	for i := 0; i < len(quotetests); i++ {
		tt := quotetests[i]
		if out := Quote(tt.in); out != tt.out {
			t.Errorf("Quote(%s) = %s, want %s", tt.in, out, tt.out)
		}
	}
}

type canBackquoteTest struct {
	in  string
	out bool
}

var canbackquotetests = []canBackquoteTest{
	{"`", false},
	{string(0), false},
	{string(1), false},
	{string(2), false},
	{string(3), false},
	{string(4), false},
	{string(5), false},
	{string(6), false},
	{string(7), false},
	{string(8), false},
	{string(9), true}, // \t
	{string(10), false},
	{string(11), false},
	{string(12), false},
	{string(13), false},
	{string(14), false},
	{string(15), false},
	{string(16), false},
	{string(17), false},
	{string(18), false},
	{string(19), false},
	{string(20), false},
	{string(21), false},
	{string(22), false},
	{string(23), false},
	{string(24), false},
	{string(25), false},
	{string(26), false},
	{string(27), false},
	{string(28), false},
	{string(29), false},
	{string(30), false},
	{string(31), false},
	{`' !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`, true},
	{`0123456789`, true},
	{`ABCDEFGHIJKLMNOPQRSTUVWXYZ`, true},
	{`abcdefghijklmnopqrstuvwxyz`, true},
	{`☺`, true},
}

func TestCanBackquote(t *testing.T) {
	for i := 0; i < len(canbackquotetests); i++ {
		tt := canbackquotetests[i]
		if out := CanBackquote(tt.in); out != tt.out {
			t.Errorf("CanBackquote(%q) = %v, want %v", tt.in, out, tt.out)
		}
	}
}

var unquotetests = []quoteTest{
	{`""`, ""},
	{`"a"`, "a"},
	{`"abc"`, "abc"},
	{`"☺"`, "☺"},
	{`"hello world"`, "hello world"},
	{`"\xFF"`, "\xFF"},
	{`"\377"`, "\377"},
	{`"\u1234"`, "\u1234"},
	{`"\U00010111"`, "\U00010111"},
	{`"\U0001011111"`, "\U0001011111"},
	{`"\a\b\f\n\r\t\v\\\""`, "\a\b\f\n\r\t\v\\\""},
	{`"'"`, "'"},

	{`'a'`, "a"},
	{`'☹'`, "☹"},
	{`'\a'`, "\a"},
	{`'\x10'`, "\x10"},
	{`'\377'`, "\377"},
	{`'\u1234'`, "\u1234"},
	{`'\U00010111'`, "\U00010111"},
	{`'\t'`, "\t"},
	{`' '`, " "},
	{`'\''`, "'"},
	{`'"'`, "\""},

	{"``", ``},
	{"`a`", `a`},
	{"`abc`", `abc`},
	{"`☺`", `☺`},
	{"`hello world`", `hello world`},
	{"`\\xFF`", `\xFF`},
	{"`\\377`", `\377`},
	{"`\\`", `\`},
	{"`	`", `	`},
	{"` `", ` `},
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
		tt := unquotetests[i]
		if out, err := Unquote(tt.in); err != nil && out != tt.out {
			t.Errorf("Unquote(%#q) = %q, %v want %q, nil", tt.in, out, err, tt.out)
		}
	}

	// run the quote tests too, backward
	for i := 0; i < len(quotetests); i++ {
		tt := quotetests[i]
		if in, err := Unquote(tt.out); in != tt.in {
			t.Errorf("Unquote(%#q) = %q, %v, want %q, nil", tt.out, in, err, tt.in)
		}
	}

	for i := 0; i < len(misquoted); i++ {
		s := misquoted[i]
		if out, err := Unquote(s); out != "" || err != os.EINVAL {
			t.Errorf("Unquote(%#q) = %q, %v want %q, %v", s, out, err, "", os.EINVAL)
		}
	}
}

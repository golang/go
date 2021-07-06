// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv_test

import (
	. "strconv"
	"strings"
	"testing"
	"unicode"
)

// Verify that our IsPrint agrees with unicode.IsPrint.
func TestIsPrint(t *testing.T) {
	n := 0
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if IsPrint(r) != unicode.IsPrint(r) {
			t.Errorf("IsPrint(%U)=%t incorrect", r, IsPrint(r))
			n++
			if n > 10 {
				return
			}
		}
	}
}

// Verify that our IsGraphic agrees with unicode.IsGraphic.
func TestIsGraphic(t *testing.T) {
	n := 0
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if IsGraphic(r) != unicode.IsGraphic(r) {
			t.Errorf("IsGraphic(%U)=%t incorrect", r, IsGraphic(r))
			n++
			if n > 10 {
				return
			}
		}
	}
}

type quoteTest struct {
	in      string
	out     string
	ascii   string
	graphic string
}

var quotetests = []quoteTest{
	{"\a\b\f\r\n\t\v", `"\a\b\f\r\n\t\v"`, `"\a\b\f\r\n\t\v"`, `"\a\b\f\r\n\t\v"`},
	{"\\", `"\\"`, `"\\"`, `"\\"`},
	{"abc\xffdef", `"abc\xffdef"`, `"abc\xffdef"`, `"abc\xffdef"`},
	{"\u263a", `"☺"`, `"\u263a"`, `"☺"`},
	{"\U0010ffff", `"\U0010ffff"`, `"\U0010ffff"`, `"\U0010ffff"`},
	{"\x04", `"\x04"`, `"\x04"`, `"\x04"`},
	// Some non-printable but graphic runes. Final column is double-quoted.
	{"!\u00a0!\u2000!\u3000!", `"!\u00a0!\u2000!\u3000!"`, `"!\u00a0!\u2000!\u3000!"`, "\"!\u00a0!\u2000!\u3000!\""},
}

func TestQuote(t *testing.T) {
	for _, tt := range quotetests {
		if out := Quote(tt.in); out != tt.out {
			t.Errorf("Quote(%s) = %s, want %s", tt.in, out, tt.out)
		}
		if out := AppendQuote([]byte("abc"), tt.in); string(out) != "abc"+tt.out {
			t.Errorf("AppendQuote(%q, %s) = %s, want %s", "abc", tt.in, out, "abc"+tt.out)
		}
	}
}

func TestQuoteToASCII(t *testing.T) {
	for _, tt := range quotetests {
		if out := QuoteToASCII(tt.in); out != tt.ascii {
			t.Errorf("QuoteToASCII(%s) = %s, want %s", tt.in, out, tt.ascii)
		}
		if out := AppendQuoteToASCII([]byte("abc"), tt.in); string(out) != "abc"+tt.ascii {
			t.Errorf("AppendQuoteToASCII(%q, %s) = %s, want %s", "abc", tt.in, out, "abc"+tt.ascii)
		}
	}
}

func TestQuoteToGraphic(t *testing.T) {
	for _, tt := range quotetests {
		if out := QuoteToGraphic(tt.in); out != tt.graphic {
			t.Errorf("QuoteToGraphic(%s) = %s, want %s", tt.in, out, tt.graphic)
		}
		if out := AppendQuoteToGraphic([]byte("abc"), tt.in); string(out) != "abc"+tt.graphic {
			t.Errorf("AppendQuoteToGraphic(%q, %s) = %s, want %s", "abc", tt.in, out, "abc"+tt.graphic)
		}
	}
}

func BenchmarkQuote(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Quote("\a\b\f\r\n\t\v\a\b\f\r\n\t\v\a\b\f\r\n\t\v")
	}
}

func BenchmarkQuoteRune(b *testing.B) {
	for i := 0; i < b.N; i++ {
		QuoteRune('\a')
	}
}

var benchQuoteBuf []byte

func BenchmarkAppendQuote(b *testing.B) {
	for i := 0; i < b.N; i++ {
		benchQuoteBuf = AppendQuote(benchQuoteBuf[:0], "\a\b\f\r\n\t\v\a\b\f\r\n\t\v\a\b\f\r\n\t\v")
	}
}

var benchQuoteRuneBuf []byte

func BenchmarkAppendQuoteRune(b *testing.B) {
	for i := 0; i < b.N; i++ {
		benchQuoteRuneBuf = AppendQuoteRune(benchQuoteRuneBuf[:0], '\a')
	}
}

type quoteRuneTest struct {
	in      rune
	out     string
	ascii   string
	graphic string
}

var quoterunetests = []quoteRuneTest{
	{'a', `'a'`, `'a'`, `'a'`},
	{'\a', `'\a'`, `'\a'`, `'\a'`},
	{'\\', `'\\'`, `'\\'`, `'\\'`},
	{0xFF, `'ÿ'`, `'\u00ff'`, `'ÿ'`},
	{0x263a, `'☺'`, `'\u263a'`, `'☺'`},
	{0xfffd, `'�'`, `'\ufffd'`, `'�'`},
	{0x0010ffff, `'\U0010ffff'`, `'\U0010ffff'`, `'\U0010ffff'`},
	{0x0010ffff + 1, `'�'`, `'\ufffd'`, `'�'`},
	{0x04, `'\x04'`, `'\x04'`, `'\x04'`},
	// Some differences between graphic and printable. Note the last column is double-quoted.
	{'\u00a0', `'\u00a0'`, `'\u00a0'`, "'\u00a0'"},
	{'\u2000', `'\u2000'`, `'\u2000'`, "'\u2000'"},
	{'\u3000', `'\u3000'`, `'\u3000'`, "'\u3000'"},
}

func TestQuoteRune(t *testing.T) {
	for _, tt := range quoterunetests {
		if out := QuoteRune(tt.in); out != tt.out {
			t.Errorf("QuoteRune(%U) = %s, want %s", tt.in, out, tt.out)
		}
		if out := AppendQuoteRune([]byte("abc"), tt.in); string(out) != "abc"+tt.out {
			t.Errorf("AppendQuoteRune(%q, %U) = %s, want %s", "abc", tt.in, out, "abc"+tt.out)
		}
	}
}

func TestQuoteRuneToASCII(t *testing.T) {
	for _, tt := range quoterunetests {
		if out := QuoteRuneToASCII(tt.in); out != tt.ascii {
			t.Errorf("QuoteRuneToASCII(%U) = %s, want %s", tt.in, out, tt.ascii)
		}
		if out := AppendQuoteRuneToASCII([]byte("abc"), tt.in); string(out) != "abc"+tt.ascii {
			t.Errorf("AppendQuoteRuneToASCII(%q, %U) = %s, want %s", "abc", tt.in, out, "abc"+tt.ascii)
		}
	}
}

func TestQuoteRuneToGraphic(t *testing.T) {
	for _, tt := range quoterunetests {
		if out := QuoteRuneToGraphic(tt.in); out != tt.graphic {
			t.Errorf("QuoteRuneToGraphic(%U) = %s, want %s", tt.in, out, tt.graphic)
		}
		if out := AppendQuoteRuneToGraphic([]byte("abc"), tt.in); string(out) != "abc"+tt.graphic {
			t.Errorf("AppendQuoteRuneToGraphic(%q, %U) = %s, want %s", "abc", tt.in, out, "abc"+tt.graphic)
		}
	}
}

type canBackquoteTest struct {
	in  string
	out bool
}

var canbackquotetests = []canBackquoteTest{
	{"`", false},
	{string(rune(0)), false},
	{string(rune(1)), false},
	{string(rune(2)), false},
	{string(rune(3)), false},
	{string(rune(4)), false},
	{string(rune(5)), false},
	{string(rune(6)), false},
	{string(rune(7)), false},
	{string(rune(8)), false},
	{string(rune(9)), true}, // \t
	{string(rune(10)), false},
	{string(rune(11)), false},
	{string(rune(12)), false},
	{string(rune(13)), false},
	{string(rune(14)), false},
	{string(rune(15)), false},
	{string(rune(16)), false},
	{string(rune(17)), false},
	{string(rune(18)), false},
	{string(rune(19)), false},
	{string(rune(20)), false},
	{string(rune(21)), false},
	{string(rune(22)), false},
	{string(rune(23)), false},
	{string(rune(24)), false},
	{string(rune(25)), false},
	{string(rune(26)), false},
	{string(rune(27)), false},
	{string(rune(28)), false},
	{string(rune(29)), false},
	{string(rune(30)), false},
	{string(rune(31)), false},
	{string(rune(0x7F)), false},
	{`' !"#$%&'()*+,-./:;<=>?@[\]^_{|}~`, true},
	{`0123456789`, true},
	{`ABCDEFGHIJKLMNOPQRSTUVWXYZ`, true},
	{`abcdefghijklmnopqrstuvwxyz`, true},
	{`☺`, true},
	{"\x80", false},
	{"a\xe0\xa0z", false},
	{"\ufeffabc", false},
	{"a\ufeffz", false},
}

func TestCanBackquote(t *testing.T) {
	for _, tt := range canbackquotetests {
		if out := CanBackquote(tt.in); out != tt.out {
			t.Errorf("CanBackquote(%q) = %v, want %v", tt.in, out, tt.out)
		}
	}
}

type unQuoteTest struct {
	in  string
	out string
}

var unquotetests = []unQuoteTest{
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
	{"`\n`", "\n"},
	{"`	`", `	`},
	{"` `", ` `},
	{"`a\rb`", "ab"},
}

var misquoted = []string{
	``,
	`"`,
	`"a`,
	`"'`,
	`b"`,
	`"\"`,
	`"\9"`,
	`"\19"`,
	`"\129"`,
	`'\'`,
	`'\9'`,
	`'\19'`,
	`'\129'`,
	`'ab'`,
	`"\x1!"`,
	`"\U12345678"`,
	`"\z"`,
	"`",
	"`xxx",
	"``x\r",
	"`\"",
	`"\'"`,
	`'\"'`,
	"\"\n\"",
	"\"\\n\n\"",
	"'\n'",
}

func TestUnquote(t *testing.T) {
	for _, tt := range unquotetests {
		testUnquote(t, tt.in, tt.out, nil)
	}
	for _, tt := range quotetests {
		testUnquote(t, tt.out, tt.in, nil)
	}
	for _, s := range misquoted {
		testUnquote(t, s, "", ErrSyntax)
	}
}

// Issue 23685: invalid UTF-8 should not go through the fast path.
func TestUnquoteInvalidUTF8(t *testing.T) {
	tests := []struct {
		in string

		// one of:
		want    string
		wantErr error
	}{
		{in: `"foo"`, want: "foo"},
		{in: `"foo`, wantErr: ErrSyntax},
		{in: `"` + "\xc0" + `"`, want: "\xef\xbf\xbd"},
		{in: `"a` + "\xc0" + `"`, want: "a\xef\xbf\xbd"},
		{in: `"\t` + "\xc0" + `"`, want: "\t\xef\xbf\xbd"},
	}
	for _, tt := range tests {
		testUnquote(t, tt.in, tt.want, tt.wantErr)
	}
}

func testUnquote(t *testing.T, in, want string, wantErr error) {
	// Test Unquote.
	got, gotErr := Unquote(in)
	if got != want || gotErr != wantErr {
		t.Errorf("Unquote(%q) = (%q, %v), want (%q, %v)", in, got, gotErr, want, wantErr)
	}

	// Test QuotedPrefix.
	// Adding an arbitrary suffix should not change the result of QuotedPrefix
	// assume that the suffix doesn't accidentally terminate a truncated input.
	if gotErr == nil {
		want = in
	}
	suffix := "\n\r\\\"`'" // special characters for quoted strings
	if len(in) > 0 {
		suffix = strings.ReplaceAll(suffix, in[:1], "")
	}
	in += suffix
	got, gotErr = QuotedPrefix(in)
	if gotErr == nil && wantErr != nil {
		_, wantErr = Unquote(got) // original input had trailing junk, reparse with only valid prefix
		want = got
	}
	if got != want || gotErr != wantErr {
		t.Errorf("QuotedPrefix(%q) = (%q, %v), want (%q, %v)", in, got, gotErr, want, wantErr)
	}
}

func BenchmarkUnquoteEasy(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Unquote(`"Give me a rock, paper and scissors and I will move the world."`)
	}
}

func BenchmarkUnquoteHard(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Unquote(`"\x47ive me a \x72ock, \x70aper and \x73cissors and \x49 will move the world."`)
	}
}

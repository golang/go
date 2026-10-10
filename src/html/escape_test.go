// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"strings"
	"testing"
)

type unescapeTest struct {
	// A short description of the test case.
	desc string
	// The HTML text.
	html string
	// The unescaped text.
	unescaped string
}

var unescapeTests = []unescapeTest{
	// Handle no entities.
	{
		"copy",
		"A\ttext\nstring",
		"A\ttext\nstring",
	},
	// Handle simple named entities.
	{
		"simple",
		"&amp; &gt; &lt;",
		"& > <",
	},
	// Handle hitting the end of the string.
	{
		"stringEnd",
		"&amp &amp",
		"& &",
	},
	// Handle entities with two codepoints.
	{
		"multiCodepoint",
		"text &gesl; blah",
		"text \u22db\ufe00 blah",
	},
	// Handle decimal numeric entities.
	{
		"decimalEntity",
		"Delta = &#916; ",
		"Delta = Δ ",
	},
	// Handle single-digit decimal numeric entities.
	{
		"singleDigitDecimalEntity",
		"Tab = &#9; = &#9 ",
		"Tab = \t = \t ",
	},
	// Handle hexadecimal numeric entities.
	{
		"hexadecimalEntity",
		"Lambda = &#x3bb; = &#X3Bb ",
		"Lambda = λ = λ ",
	},
	// Handle single-digit hexadecimal numeric entities.
	{
		"singleDigitHexadecimalEntity",
		"Tab = &#x9; = &#x9 ",
		"Tab = \t = \t ",
	},
	// Handle numeric early termination.
	{
		"numericEnds",
		"&# &#x &#128;43 &copy = &#169f = &#xa9",
		"&# &#x €43 © = ©f = ©",
	},
	// Handle numeric ISO-8859-1 entity replacements.
	{
		"numericReplacements",
		"Footnote&#x87;",
		"Footnote‡",
	},
	// Handle single ampersand.
	{
		"copySingleAmpersand",
		"&",
		"&",
	},
	// Handle ampersand followed by non-entity.
	{
		"copyAmpersandNonEntity",
		"text &test",
		"text &test",
	},
	// Handle "&#".
	{
		"copyAmpersandHash",
		"text &#",
		"text &#",
	},
}

func TestUnescape(t *testing.T) {
	for _, tt := range unescapeTests {
		unescaped := UnescapeString(tt.html)
		if unescaped != tt.unescaped {
			t.Errorf("TestUnescape %s: want %q, got %q", tt.desc, tt.unescaped, unescaped)
		}
	}
}

func TestUnescapeEscape(t *testing.T) {
	ss := []string{
		``,
		`abc def`,
		`a & b`,
		`a&amp;b`,
		`a &amp b`,
		`&quot;`,
		`"`,
		`"<&>"`,
		`&quot;&lt;&amp;&gt;&quot;`,
		`3&5==1 && 0<1, "0&lt;1", a+acute=&aacute;`,
		`The special characters are: <, >, &, ' and "`,
		`&#9; &#9 &#x9; &#x9`,
	}
	for _, s := range ss {
		if got := UnescapeString(EscapeString(s)); got != s {
			t.Errorf("got %q want %q", got, s)
		}
	}
}

var (
	benchEscapeData     = strings.Repeat("AAAAA < BBBBB > CCCCC & DDDDD ' EEEEE \" ", 100)
	benchEscapeNone     = strings.Repeat("AAAAA x BBBBB x CCCCC x DDDDD x EEEEE x ", 100)
	benchUnescapeSparse = strings.Repeat(strings.Repeat("AAAAA x BBBBB x CCCCC x DDDDD x EEEEE x ", 10)+"&amp;", 10)
	benchUnescapeDense  = strings.Repeat("&amp;&lt; &amp; &lt;", 100)
)

func BenchmarkEscape(b *testing.B) {
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(EscapeString(benchEscapeData))
	}
}

func BenchmarkEscapeNone(b *testing.B) {
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(EscapeString(benchEscapeNone))
	}
}

func BenchmarkUnescape(b *testing.B) {
	s := EscapeString(benchEscapeData)
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(UnescapeString(s))
	}
}

func BenchmarkUnescapeNone(b *testing.B) {
	s := EscapeString(benchEscapeNone)
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(UnescapeString(s))
	}
}

func BenchmarkUnescapeSparse(b *testing.B) {
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(UnescapeString(benchUnescapeSparse))
	}
}

func BenchmarkUnescapeDense(b *testing.B) {
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(UnescapeString(benchUnescapeDense))
	}
}

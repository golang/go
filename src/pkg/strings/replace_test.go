// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	"bytes"
	"fmt"
	"log"
	. "strings"
	"testing"
)

var _ = log.Printf

type ReplacerTest struct {
	r   *Replacer
	in  string
	out string
}

var htmlEscaper = NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", "\"", "&quot;")

// The http package's old HTML escaping function.
func oldhtmlEscape(s string) string {
	s = Replace(s, "&", "&amp;", -1)
	s = Replace(s, "<", "&lt;", -1)
	s = Replace(s, ">", "&gt;", -1)
	s = Replace(s, "\"", "&quot;", -1)
	s = Replace(s, "'", "&apos;", -1)
	return s
}

var replacer = NewReplacer("aaa", "3[aaa]", "aa", "2[aa]", "a", "1[a]", "i", "i",
	"longerst", "most long", "longer", "medium", "long", "short",
	"X", "Y", "Y", "Z")

var capitalLetters = NewReplacer("a", "A", "b", "B")

var blankToXReplacer = NewReplacer("", "X", "o", "O")

var ReplacerTests = []ReplacerTest{
	// byte->string
	{htmlEscaper, "No changes", "No changes"},
	{htmlEscaper, "I <3 escaping & stuff", "I &lt;3 escaping &amp; stuff"},
	{htmlEscaper, "&&&", "&amp;&amp;&amp;"},

	// generic
	{replacer, "fooaaabar", "foo3[aaa]b1[a]r"},
	{replacer, "long, longerst, longer", "short, most long, medium"},
	{replacer, "XiX", "YiY"},

	// byte->byte
	{capitalLetters, "brad", "BrAd"},
	{capitalLetters, Repeat("a", (32<<10)+123), Repeat("A", (32<<10)+123)},

	// hitting "" special case
	{blankToXReplacer, "oo", "XOXOX"},
}

func TestReplacer(t *testing.T) {
	for i, tt := range ReplacerTests {
		if s := tt.r.Replace(tt.in); s != tt.out {
			t.Errorf("%d. Replace(%q) = %q, want %q", i, tt.in, s, tt.out)
		}
		var buf bytes.Buffer
		n, err := tt.r.WriteString(&buf, tt.in)
		if err != nil {
			t.Errorf("%d. WriteString: %v", i, err)
			continue
		}
		got := buf.String()
		if got != tt.out {
			t.Errorf("%d. WriteString(%q) wrote %q, want %q", i, tt.in, got, tt.out)
			continue
		}
		if n != len(tt.out) {
			t.Errorf("%d. WriteString(%q) wrote correct string but reported %d bytes; want %d (%q)",
				i, tt.in, n, len(tt.out), tt.out)
		}
	}
}

// pickAlgorithmTest is a test that verifies that given input for a
// Replacer that we pick the correct algorithm.
type pickAlgorithmTest struct {
	r    *Replacer
	want string // name of algorithm
}

var pickAlgorithmTests = []pickAlgorithmTest{
	{capitalLetters, "*strings.byteReplacer"},
	{NewReplacer("12", "123"), "*strings.genericReplacer"},
	{NewReplacer("1", "12"), "*strings.byteStringReplacer"},
	{htmlEscaper, "*strings.byteStringReplacer"},
}

func TestPickAlgorithm(t *testing.T) {
	for i, tt := range pickAlgorithmTests {
		got := fmt.Sprintf("%T", tt.r.Replacer())
		if got != tt.want {
			t.Errorf("%d. algorithm = %s, want %s", i, got, tt.want)
		}
	}
}

func BenchmarkGenericMatch(b *testing.B) {
	str := Repeat("A", 100) + Repeat("B", 100)
	generic := NewReplacer("a", "A", "b", "B", "12", "123") // varying lengths forces generic
	for i := 0; i < b.N; i++ {
		generic.Replace(str)
	}
}

func BenchmarkByteByteNoMatch(b *testing.B) {
	str := Repeat("A", 100) + Repeat("B", 100)
	for i := 0; i < b.N; i++ {
		capitalLetters.Replace(str)
	}
}

func BenchmarkByteByteMatch(b *testing.B) {
	str := Repeat("a", 100) + Repeat("b", 100)
	for i := 0; i < b.N; i++ {
		capitalLetters.Replace(str)
	}
}

func BenchmarkByteStringMatch(b *testing.B) {
	str := "<" + Repeat("a", 99) + Repeat("b", 99) + ">"
	for i := 0; i < b.N; i++ {
		htmlEscaper.Replace(str)
	}
}

func BenchmarkHTMLEscapeNew(b *testing.B) {
	str := "I <3 to escape HTML & other text too."
	for i := 0; i < b.N; i++ {
		htmlEscaper.Replace(str)
	}
}

func BenchmarkHTMLEscapeOld(b *testing.B) {
	str := "I <3 to escape HTML & other text too."
	for i := 0; i < b.N; i++ {
		oldhtmlEscape(str)
	}
}

// BenchmarkByteByteReplaces compares byteByteImpl against multiple Replaces.
func BenchmarkByteByteReplaces(b *testing.B) {
	str := Repeat("a", 100) + Repeat("b", 100)
	for i := 0; i < b.N; i++ {
		Replace(Replace(str, "a", "A", -1), "b", "B", -1)
	}
}

// BenchmarkByteByteMap compares byteByteImpl against Map.
func BenchmarkByteByteMap(b *testing.B) {
	str := Repeat("a", 100) + Repeat("b", 100)
	fn := func(r rune) rune {
		switch r {
		case 'a':
			return 'A'
		case 'b':
			return 'B'
		}
		return r
	}
	for i := 0; i < b.N; i++ {
		Map(fn, str)
	}
}

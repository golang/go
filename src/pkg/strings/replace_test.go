// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings_test

import (
	. "strings"
	"testing"
)

type ReplacerTest struct {
	m   *Replacer
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

var ReplacerTests = []ReplacerTest{
	{htmlEscaper, "No changes", "No changes"},
	{htmlEscaper, "I <3 escaping & stuff", "I &lt;3 escaping &amp; stuff"},
	{htmlEscaper, "&&&", "&amp;&amp;&amp;"},
	{replacer, "fooaaabar", "foo3[aaa]b1[a]r"},
	{replacer, "long, longerst, longer", "short, most long, medium"},
	{replacer, "XiX", "YiY"},
}

func TestReplacer(t *testing.T) {
	for i, tt := range ReplacerTests {
		if s := tt.m.Replace(tt.in); s != tt.out {
			t.Errorf("%d. Replace(%q) = %q, want %q", i, tt.in, s, tt.out)
		}
	}
}

var slowReplacer = NewReplacer("&&", "&amp;", "<<", "&lt;", ">>", "&gt;", "\"\"", "&quot;", "''", "&apos;")

func BenchmarkReplacerSingleByte(b *testing.B) {
	str := "I <3 benchmarking html & other stuff too >:D"
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(htmlEscaper.Replace(str))
	}
}

func BenchmarkReplaceMap(b *testing.B) {
	str := "I <<3 benchmarking html && other stuff too >>:D"
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(slowReplacer.Replace(str))
	}
}

func BenchmarkOldHTTPHTMLReplace(b *testing.B) {
	str := "I <3 benchmarking html & other stuff too >:D"
	n := 0
	for i := 0; i < b.N; i++ {
		n += len(oldhtmlEscape(str))
	}
}

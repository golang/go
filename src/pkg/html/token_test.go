// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"os"
	"testing"
)

type tokenTest struct {
	// A short description of the test case.
	desc string
	// The HTML to parse.
	html string
	// The string representations of the expected tokens.
	tokens []string
}

var tokenTests = []tokenTest{
	// A single text node. The tokenizer should not break text nodes on whitespace,
	// nor should it normalize whitespace within a text node.
	{
		"text",
		"foo  bar",
		[]string{
			"foo  bar",
		},
	},
	// An entity.
	{
		"entity",
		"one &lt; two",
		[]string{
			"one &lt; two",
		},
	},
	// A start, self-closing and end tag. The tokenizer does not care if the start
	// and end tokens don't match; that is the job of the parser.
	{
		"tags",
		"<a>b<c/>d</e>",
		[]string{
			"<a>",
			"b",
			"<c/>",
			"d",
			"</e>",
		},
	},
	// An attribute with a backslash.
	{
		"backslash",
		`<p id="a\"b">`,
		[]string{
			`<p id="a&quot;b">`,
		},
	},
	// Entities, tag name and attribute key lower-casing, and whitespace
	// normalization within a tag.
	{
		"tricky",
		"<p \t\n iD=\"a&quot;B\"  foo=\"bar\"><EM>te&lt;&amp;;xt</em></p>",
		[]string{
			`<p id="a&quot;B" foo="bar">`,
			"<em>",
			"te&lt;&amp;;xt",
			"</em>",
			"</p>",
		},
	},
	// A non-existant entity. Tokenizing and converting back to a string should
	// escape the "&" to become "&amp;".
	{
		"noSuchEntity",
		`<a b="c&noSuchEntity;d">&lt;&alsoDoesntExist;&`,
		[]string{
			`<a b="c&amp;noSuchEntity;d">`,
			"&lt;&amp;alsoDoesntExist;&amp;",
		},
	},
}

func TestTokenizer(t *testing.T) {
loop:
	for _, tt := range tokenTests {
		z := NewTokenizer(bytes.NewBuffer([]byte(tt.html)))
		for i, s := range tt.tokens {
			if z.Next() == Error {
				t.Errorf("%s token %d: want %q got error %v", tt.desc, i, s, z.Error())
				continue loop
			}
			actual := z.Token().String()
			if s != actual {
				t.Errorf("%s token %d: want %q got %q", tt.desc, i, s, actual)
				continue loop
			}
		}
		z.Next()
		if z.Error() != os.EOF {
			t.Errorf("%s: want EOF got %q", tt.desc, z.Token().String())
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
	}
	for _, s := range ss {
		if s != UnescapeString(EscapeString(s)) {
			t.Errorf("s != UnescapeString(EscapeString(s)), s=%q", s)
		}
	}
}

func TestBufAPI(t *testing.T) {
	s := "0<a>1</a>2<b>3<a>4<a>5</a>6</b>7</a>8<a/>9"
	z := NewTokenizer(bytes.NewBuffer([]byte(s)))
	result := bytes.NewBuffer(nil)
	depth := 0
loop:
	for {
		tt := z.Next()
		switch tt {
		case Error:
			if z.Error() != os.EOF {
				t.Error(z.Error())
			}
			break loop
		case Text:
			if depth > 0 {
				result.Write(z.Text())
			}
		case StartTag, EndTag:
			tn, _ := z.TagName()
			if len(tn) == 1 && tn[0] == 'a' {
				if tt == StartTag {
					depth++
				} else {
					depth--
				}
			}
		}
	}
	u := "14567"
	v := string(result.Bytes())
	if u != v {
		t.Errorf("TestBufAPI: want %q got %q", u, v)
	}
}

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

type tokenTest struct {
	// A short description of the test case.
	desc string
	// The HTML to parse.
	html string
	// The string representations of the expected tokens, joined by '$'.
	golden string
}

var tokenTests = []tokenTest{
	{
		"empty",
		"",
		"",
	},
	// A single text node. The tokenizer should not break text nodes on whitespace,
	// nor should it normalize whitespace within a text node.
	{
		"text",
		"foo  bar",
		"foo  bar",
	},
	// An entity.
	{
		"entity",
		"one &lt; two",
		"one &lt; two",
	},
	// A start, self-closing and end tag. The tokenizer does not care if the start
	// and end tokens don't match; that is the job of the parser.
	{
		"tags",
		"<a>b<c/>d</e>",
		"<a>$b$<c/>$d$</e>",
	},
	// Angle brackets that aren't a tag.
	{
		"not a tag #0",
		"<",
		"&lt;",
	},
	{
		"not a tag #1",
		"</",
		"&lt;/",
	},
	{
		"not a tag #2",
		"</>",
		"",
	},
	{
		"not a tag #3",
		"a</>b",
		"a$b",
	},
	{
		"not a tag #4",
		"</ >",
		"<!-- -->",
	},
	{
		"not a tag #5",
		"</.",
		"<!--.-->",
	},
	{
		"not a tag #6",
		"</.>",
		"<!--.-->",
	},
	{
		"not a tag #7",
		"a < b",
		"a &lt; b",
	},
	{
		"not a tag #8",
		"<.>",
		"&lt;.&gt;",
	},
	{
		"not a tag #9",
		"a<<<b>>>c",
		"a&lt;&lt;$<b>$&gt;&gt;c",
	},
	{
		"not a tag #10",
		"if x<0 and y < 0 then x*y>0",
		"if x&lt;0 and y &lt; 0 then x*y&gt;0",
	},
	// EOF in a tag name.
	{
		"tag name eof #0",
		"<a",
		"",
	},
	{
		"tag name eof #1",
		"<a ",
		"",
	},
	{
		"tag name eof #2",
		"a<b",
		"a",
	},
	{
		"tag name eof #3",
		"<a><b",
		"<a>",
	},
	{
		"tag name eof #4",
		`<a x`,
		`<a x="">`,
	},
	// Some malformed tags that are missing a '>'.
	{
		"malformed tag #0",
		`<p</p>`,
		`<p< p="">`,
	},
	{
		"malformed tag #1",
		`<p </p>`,
		`<p <="" p="">`,
	},
	{
		"malformed tag #2",
		`<p id`,
		`<p id="">`,
	},
	{
		"malformed tag #3",
		`<p id=`,
		`<p id="">`,
	},
	{
		"malformed tag #4",
		`<p id=>`,
		`<p id="">`,
	},
	{
		"malformed tag #5",
		`<p id=0`,
		`<p id="0">`,
	},
	{
		"malformed tag #6",
		`<p id=0</p>`,
		`<p id="0&lt;/p">`,
	},
	{
		"malformed tag #7",
		`<p id="0</p>`,
		`<p id="0&lt;/p&gt;">`,
	},
	{
		"malformed tag #8",
		`<p id="0"</p>`,
		`<p id="0" <="" p="">`,
	},
	// Raw text and RCDATA.
	{
		"basic raw text",
		"<script><a></b></script>",
		"<script>$&lt;a&gt;&lt;/b&gt;$</script>",
	},
	{
		"unfinished script end tag",
		"<SCRIPT>a</SCR",
		"<script>$a&lt;/SCR",
	},
	{
		"broken script end tag",
		"<SCRIPT>a</SCR ipt>",
		"<script>$a&lt;/SCR ipt&gt;",
	},
	{
		"EOF in script end tag",
		"<SCRIPT>a</SCRipt",
		"<script>$a&lt;/SCRipt",
	},
	{
		"scriptx end tag",
		"<SCRIPT>a</SCRiptx",
		"<script>$a&lt;/SCRiptx",
	},
	{
		"' ' completes script end tag",
		"<SCRIPT>a</SCRipt ",
		"<script>$a$</script>",
	},
	{
		"'>' completes script end tag",
		"<SCRIPT>a</SCRipt>",
		"<script>$a$</script>",
	},
	{
		"self-closing script end tag",
		"<SCRIPT>a</SCRipt/>",
		"<script>$a$</script>",
	},
	{
		"nested script tag",
		"<SCRIPT>a</SCRipt<script>",
		"<script>$a&lt;/SCRipt&lt;script&gt;",
	},
	{
		"script end tag after unfinished",
		"<SCRIPT>a</SCRipt</script>",
		"<script>$a&lt;/SCRipt$</script>",
	},
	{
		"script/style mismatched tags",
		"<script>a</style>",
		"<script>$a&lt;/style&gt;",
	},
	{
		"style element with entity",
		"<style>&apos;",
		"<style>$&amp;apos;",
	},
	{
		"textarea with tag",
		"<textarea><div></textarea>",
		"<textarea>$&lt;div&gt;$</textarea>",
	},
	{
		"title with tag and entity",
		"<title><b>K&amp;R C</b></title>",
		"<title>$&lt;b&gt;K&amp;R C&lt;/b&gt;$</title>",
	},
	// DOCTYPE tests.
	{
		"Proper DOCTYPE",
		"<!DOCTYPE html>",
		"<!DOCTYPE html>",
	},
	{
		"DOCTYPE with no space",
		"<!doctypehtml>",
		"<!DOCTYPE html>",
	},
	{
		"DOCTYPE with two spaces",
		"<!doctype  html>",
		"<!DOCTYPE html>",
	},
	{
		"looks like DOCTYPE but isn't",
		"<!DOCUMENT html>",
		"<!--DOCUMENT html-->",
	},
	{
		"DOCTYPE at EOF",
		"<!DOCtype",
		"<!DOCTYPE >",
	},
	// XML processing instructions.
	{
		"XML processing instruction",
		"<?xml?>",
		"<!--?xml?-->",
	},
	// Comments.
	{
		"comment0",
		"abc<b><!-- skipme --></b>def",
		"abc$<b>$<!-- skipme -->$</b>$def",
	},
	{
		"comment1",
		"a<!-->z",
		"a$<!---->$z",
	},
	{
		"comment2",
		"a<!--->z",
		"a$<!---->$z",
	},
	{
		"comment3",
		"a<!--x>-->z",
		"a$<!--x>-->$z",
	},
	{
		"comment4",
		"a<!--x->-->z",
		"a$<!--x->-->$z",
	},
	{
		"comment5",
		"a<!>z",
		"a$<!---->$z",
	},
	{
		"comment6",
		"a<!->z",
		"a$<!----->$z",
	},
	{
		"comment7",
		"a<!---<>z",
		"a$<!---<>z-->",
	},
	{
		"comment8",
		"a<!--z",
		"a$<!--z-->",
	},
	{
		"comment9",
		"a<!--x--!>z",
		"a$<!--x-->$z",
	},
	// An attribute with a backslash.
	{
		"backslash",
		`<p id="a\"b">`,
		`<p id="a\" b"="">`,
	},
	// Entities, tag name and attribute key lower-casing, and whitespace
	// normalization within a tag.
	{
		"tricky",
		"<p \t\n iD=\"a&quot;B\"  foo=\"bar\"><EM>te&lt;&amp;;xt</em></p>",
		`<p id="a&quot;B" foo="bar">$<em>$te&lt;&amp;;xt$</em>$</p>`,
	},
	// A nonexistent entity. Tokenizing and converting back to a string should
	// escape the "&" to become "&amp;".
	{
		"noSuchEntity",
		`<a b="c&noSuchEntity;d">&lt;&alsoDoesntExist;&`,
		`<a b="c&amp;noSuchEntity;d">$&lt;&amp;alsoDoesntExist;&amp;`,
	},
	/*
		// TODO: re-enable this test when it works. This input/output matches html5lib's behavior.
		{
			"entity without semicolon",
			`&notit;&notin;<a b="q=z&amp=5&notice=hello&not;=world">`,
			`¬it;∉$<a b="q=z&amp;amp=5&amp;notice=hello¬=world">`,
		},
	*/
	{
		"entity with digits",
		"&frac12;",
		"½",
	},
	// Attribute tests:
	// http://dev.w3.org/html5/spec/Overview.html#attributes-0
	{
		"Empty attribute",
		`<input disabled FOO>`,
		`<input disabled="" foo="">`,
	},
	{
		"Empty attribute, whitespace",
		`<input disabled FOO >`,
		`<input disabled="" foo="">`,
	},
	{
		"Unquoted attribute value",
		`<input value=yes FOO=BAR>`,
		`<input value="yes" foo="BAR">`,
	},
	{
		"Unquoted attribute value, spaces",
		`<input value = yes FOO = BAR>`,
		`<input value="yes" foo="BAR">`,
	},
	{
		"Unquoted attribute value, trailing space",
		`<input value=yes FOO=BAR >`,
		`<input value="yes" foo="BAR">`,
	},
	{
		"Single-quoted attribute value",
		`<input value='yes' FOO='BAR'>`,
		`<input value="yes" foo="BAR">`,
	},
	{
		"Single-quoted attribute value, trailing space",
		`<input value='yes' FOO='BAR' >`,
		`<input value="yes" foo="BAR">`,
	},
	{
		"Double-quoted attribute value",
		`<input value="I'm an attribute" FOO="BAR">`,
		`<input value="I&apos;m an attribute" foo="BAR">`,
	},
	{
		"Attribute name characters",
		`<meta http-equiv="content-type">`,
		`<meta http-equiv="content-type">`,
	},
	{
		"Mixed attributes",
		`a<P V="0 1" w='2' X=3 y>z`,
		`a$<p v="0 1" w="2" x="3" y="">$z`,
	},
	{
		"Attributes with a solitary single quote",
		`<p id=can't><p id=won't>`,
		`<p id="can&apos;t">$<p id="won&apos;t">`,
	},
}

func TestTokenizer(t *testing.T) {
loop:
	for _, tt := range tokenTests {
		z := NewTokenizer(strings.NewReader(tt.html))
		if tt.golden != "" {
			for i, s := range strings.Split(tt.golden, "$") {
				if z.Next() == ErrorToken {
					t.Errorf("%s token %d: want %q got error %v", tt.desc, i, s, z.Error())
					continue loop
				}
				actual := z.Token().String()
				if s != actual {
					t.Errorf("%s token %d: want %q got %q", tt.desc, i, s, actual)
					continue loop
				}
			}
		}
		z.Next()
		if z.Error() != io.EOF {
			t.Errorf("%s: want EOF got %q", tt.desc, z.Error())
		}
	}
}

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
	// Handle hexadecimal numeric entities.
	{
		"hexadecimalEntity",
		"Lambda = &#x3bb; = &#X3Bb ",
		"Lambda = λ = λ ",
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
		case ErrorToken:
			if z.Error() != io.EOF {
				t.Error(z.Error())
			}
			break loop
		case TextToken:
			if depth > 0 {
				result.Write(z.Text())
			}
		case StartTagToken, EndTagToken:
			tn, _ := z.TagName()
			if len(tn) == 1 && tn[0] == 'a' {
				if tt == StartTagToken {
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

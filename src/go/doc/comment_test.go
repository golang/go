// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"bytes"
	"io"
	"reflect"
	"strings"
	"testing"
	"text/template"
)

// This is practically a htmlFormatter, this is because it uses a lot of the possible things
// a Formatter can do. This makes it easy to use for testing
type testFormatter struct {
	out    io.Writer
	headID string
}

// Escape escapes text for HTML. If nice is set,
// also turn `` and '' into appropirate quotes.
func (f *testFormatter) Escape(text string, nice bool) {
	if nice {
		// In the first pass, we convert `` and '' into their unicode equivalents.
		// This prevents them from being escaped in HTMLEscape.
		text = convertQuotes(text)
		var buf bytes.Buffer
		template.HTMLEscape(&buf, []byte(text))
		// Now we convert the unicode quotes to their HTML escaped entities to maintain old behavior.
		// We need to use a temp buffer to read the string back and do the conversion,
		// otherwise HTMLEscape will escape & to &amp;
		htmlQuoteReplacer.WriteString(f.out, buf.String())
		return
	}
	template.HTMLEscape(f.out, []byte(text))
}

func (f *testFormatter) WriteURL(url, match string, italics, nice bool) {
	if len(url) > 0 {
		f.out.Write(htmlPreLink)
		f.Escape(url, false)
		f.out.Write(htmlPostLink)
	}
	if italics {
		f.out.Write(htmlStartI)
	}
	f.Escape(match, nice)
	if italics {
		f.out.Write(htmlEndI)
	}
	if len(url) > 0 {
		f.out.Write(htmlEndLink)
	}
}

func (f *testFormatter) StartPara() {
	f.out.Write(htmlStartP)
}

func (f *testFormatter) PreParaLine(line string)  {}
func (f *testFormatter) PostParaLine(line string) {}

func (f *testFormatter) EndPara() {
	f.out.Write(htmlEndP)
}

func (f *testFormatter) StartHead() {
	f.out.Write(htmlPreH)
	f.headID = ""
}

func (f *testFormatter) PreHeadLine(line string) {
	if f.headID == "" {
		f.headID = anchorID(line)
		f.out.Write([]byte(f.headID))
		f.out.Write(htmlPostH)
	}
}

func (f *testFormatter) PostHeadLine(line string) {}

func (f *testFormatter) EndHead() {
	if f.headID == "" {
		f.out.Write(htmlPostH)
	}
	f.out.Write(htmlEndH)
}

func (f *testFormatter) StartRaw() {
	f.out.Write(htmlStartPre)
}

func (f *testFormatter) PreRawLine(line string)  {}
func (f *testFormatter) PostRawLine(line string) {}

func (f *testFormatter) EndRaw() {
	f.out.Write(htmlEndPre)
}

var headingTests = []struct {
	line string
	ok   bool
}{
	{"Section", true},
	{"A typical usage", true},
	{"ΔΛΞ is Greek", true},
	{"Foo 42", true},
	{"", false},
	{"section", false},
	{"A typical usage:", false},
	{"This code:", false},
	{"δ is Greek", false},
	{"Foo §", false},
	{"Fermat's Last Sentence", true},
	{"Fermat's", true},
	{"'sX", false},
	{"Ted 'Too' Bar", false},
	{"Use n+m", false},
	{"Scanning:", false},
	{"N:M", false},
}

func TestIsHeading(t *testing.T) {
	for _, tt := range headingTests {
		if h := heading(tt.line); (len(h) > 0) != tt.ok {
			t.Errorf("isHeading(%q) = %v, want %v", tt.line, h, tt.ok)
		}
	}
}

var blocksTests = []struct {
	in   string
	out  []block
	text string
}{
	{
		in: `Para 1.
Para 1 line 2.

Para 2.

Section

Para 3.

	pre
	pre1

Para 4.

	pre
	pre1

	pre2

Para 5.


	pre


	pre1
	pre2

Para 6.
	pre
	pre2
`,
		out: []block{
			{opPara, []string{"Para 1.\n", "Para 1 line 2.\n"}},
			{opPara, []string{"Para 2.\n"}},
			{opHead, []string{"Section"}},
			{opPara, []string{"Para 3.\n"}},
			{opRaw, []string{"pre\n", "pre1\n"}},
			{opPara, []string{"Para 4.\n"}},
			{opRaw, []string{"pre\n", "pre1\n", "\n", "pre2\n"}},
			{opPara, []string{"Para 5.\n"}},
			{opRaw, []string{"pre\n", "\n", "\n", "pre1\n", "pre2\n"}},
			{opPara, []string{"Para 6.\n"}},
			{opRaw, []string{"pre\n", "pre2\n"}},
		},
		text: `.   Para 1. Para 1 line 2.

.   Para 2.


.   Section

.   Para 3.

$	pre
$	pre1

.   Para 4.

$	pre
$	pre1

$	pre2

.   Para 5.

$	pre


$	pre1
$	pre2

.   Para 6.

$	pre
$	pre2
`,
	},
	{
		in: "Para.\n\tshould not be ``escaped''",
		out: []block{
			{opPara, []string{"Para.\n"}},
			{opRaw, []string{"should not be ``escaped''"}},
		},
		text: ".   Para.\n\n$	should not be ``escaped''",
	},
	{
		in: "// A very long line of 46 char for line wrapping.",
		out: []block{
			{opPara, []string{"// A very long line of 46 char for line wrapping."}},
		},
		text: `.   // A very long line of 46 char for line
.   // wrapping.
`,
	},
	{
		in: `/* A very long line of 46 char for line wrapping.
A very long line of 46 char for line wrapping. */`,
		out: []block{
			{opPara, []string{"/* A very long line of 46 char for line wrapping.\n", "A very long line of 46 char for line wrapping. */"}},
		},
		text: `.   /* A very long line of 46 char for line
.   wrapping. A very long line of 46 char
.   for line wrapping. */
`,
	},
}

func TestBlocks(t *testing.T) {
	for i, tt := range blocksTests {
		b := blocks(tt.in)
		if !reflect.DeepEqual(b, tt.out) {
			t.Errorf("#%d: mismatch\nhave: %v\nwant: %v", i, b, tt.out)
		}
	}
}

func TestToText(t *testing.T) {
	var buf bytes.Buffer
	for i, tt := range blocksTests {
		ToText(&buf, tt.in, ".   ", "$\t", 40)
		if have := buf.String(); have != tt.text {
			t.Errorf("#%d: mismatch\nhave: %s\nwant: %s\nhave vs want:\n%q\n%q", i, have, tt.text, have, tt.text)
		}
		buf.Reset()
	}
}

var emphasizeTests = []struct {
	in, out string
}{
	{"", ""},
	{"http://[::1]:8080/foo.txt", `<a href="http://[::1]:8080/foo.txt">http://[::1]:8080/foo.txt</a>`},
	{"before (https://www.google.com) after", `before (<a href="https://www.google.com">https://www.google.com</a>) after`},
	{"before https://www.google.com:30/x/y/z:b::c. After", `before <a href="https://www.google.com:30/x/y/z:b::c">https://www.google.com:30/x/y/z:b::c</a>. After`},
	{"http://www.google.com/path/:;!-/?query=%34b#093124", `<a href="http://www.google.com/path/:;!-/?query=%34b#093124">http://www.google.com/path/:;!-/?query=%34b#093124</a>`},
	{"http://www.google.com/path/:;!-/?query=%34bar#093124", `<a href="http://www.google.com/path/:;!-/?query=%34bar#093124">http://www.google.com/path/:;!-/?query=%34bar#093124</a>`},
	{"http://www.google.com/index.html! After", `<a href="http://www.google.com/index.html">http://www.google.com/index.html</a>! After`},
	{"http://www.google.com/", `<a href="http://www.google.com/">http://www.google.com/</a>`},
	{"https://www.google.com/", `<a href="https://www.google.com/">https://www.google.com/</a>`},
	{"http://www.google.com/path.", `<a href="http://www.google.com/path">http://www.google.com/path</a>.`},
	{"http://en.wikipedia.org/wiki/Camellia_(cipher)", `<a href="http://en.wikipedia.org/wiki/Camellia_(cipher)">http://en.wikipedia.org/wiki/Camellia_(cipher)</a>`},
	{"(http://www.google.com/)", `(<a href="http://www.google.com/">http://www.google.com/</a>)`},
	{"http://gmail.com)", `<a href="http://gmail.com">http://gmail.com</a>)`},
	{"((http://gmail.com))", `((<a href="http://gmail.com">http://gmail.com</a>))`},
	{"http://gmail.com ((http://gmail.com)) ()", `<a href="http://gmail.com">http://gmail.com</a> ((<a href="http://gmail.com">http://gmail.com</a>)) ()`},
	{"Foo bar http://example.com/ quux!", `Foo bar <a href="http://example.com/">http://example.com/</a> quux!`},
	{"Hello http://example.com/%2f/ /world.", `Hello <a href="http://example.com/%2f/">http://example.com/%2f/</a> /world.`},
	{"Lorem http: ipsum //host/path", "Lorem http: ipsum //host/path"},
	{"javascript://is/not/linked", "javascript://is/not/linked"},
	{"http://foo", `<a href="http://foo">http://foo</a>`},
	{"art by [[https://www.example.com/person/][Person Name]]", `art by [[<a href="https://www.example.com/person/">https://www.example.com/person/</a>][Person Name]]`},
	{"please visit (http://golang.org/)", `please visit (<a href="http://golang.org/">http://golang.org/</a>)`},
	{"please visit http://golang.org/hello())", `please visit <a href="http://golang.org/hello()">http://golang.org/hello()</a>)`},
	{"http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD", `<a href="http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD">http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD</a>`},
	{"https://foo.bar/bal/x(])", `<a href="https://foo.bar/bal/x(">https://foo.bar/bal/x(</a>])`}, // inner ] causes (]) to be cut off from URL
	{"foo [ http://bar(])", `foo [ <a href="http://bar(">http://bar(</a>])`},                      // outer [ causes ]) to be cut off from URL
}

func TestEmphasize(t *testing.T) {
	f := &testFormatter{}
	for i, tt := range emphasizeTests {
		var buf bytes.Buffer
		f.out = &buf
		emphasize(&buf, f, tt.in, nil, true)
		out := buf.String()
		if out != tt.out {
			t.Errorf("#%d: mismatch\nhave: %v\nwant: %v", i, out, tt.out)
		}
	}
}

func TestHtmlEscape(t *testing.T) {
	html := &htmlFormatter{}
	commentTests := []struct {
		in, out string
	}{
		{"typically invoked as ``go tool asm'',", "typically invoked as " + ldquo + "go tool asm" + rdquo + ","},
		{"For more detail, run ``go help test'' and ``go help testflag''", "For more detail, run " + ldquo + "go help test" + rdquo + " and " + ldquo + "go help testflag" + rdquo},
	}
	for i, tt := range commentTests {
		var buf strings.Builder
		html.out = &buf
		html.Escape(tt.in, true)
		out := buf.String()
		if out != tt.out {
			t.Errorf("#%d: mismatch\nhave: %q\nwant: %q", i, out, tt.out)
		}
	}
}

func TestHtmlWriteURL(t *testing.T) {
	html := &htmlFormatter{}
	commentTests := []struct {
		url, match, out string
	}{
		{"http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD", "qemu git repo", "<a href=\"http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD\"><i>qemu git repo</i></a>"},
		{"http://gmail.com", "gmail", "<a href=\"http://gmail.com\"><i>gmail</i></a>"},
	}
	for i, tt := range commentTests {
		var buf strings.Builder
		html.out = &buf
		html.WriteURL(tt.url, tt.match, true, true)
		out := buf.String()
		if out != tt.out {
			t.Errorf("#%d: mismatch\nhave: %q\nwant: %q", i, out, tt.out)
		}
	}
}

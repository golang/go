// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package doc

import (
	"bytes"
	"reflect"
	"testing"
)

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
			{opPre, []string{"pre\n", "pre1\n"}},
			{opPara, []string{"Para 4.\n"}},
			{opPre, []string{"pre\n", "pre1\n", "\n", "pre2\n"}},
			{opPara, []string{"Para 5.\n"}},
			{opPre, []string{"pre\n", "\n", "\n", "pre1\n", "pre2\n"}},
			{opPara, []string{"Para 6.\n"}},
			{opPre, []string{"pre\n", "pre2\n"}},
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
	{"http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD", `<a href="http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD">http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD</a>`},
}

func TestEmphasize(t *testing.T) {
	for i, tt := range emphasizeTests {
		var buf bytes.Buffer
		emphasize(&buf, tt.in, nil, true)
		out := buf.String()
		if out != tt.out {
			t.Errorf("#%d: mismatch\nhave: %v\nwant: %v", i, out, tt.out)
		}
	}
}

var pairedParensPrefixLenTests = []struct {
	in, out string
}{
	{"", ""},
	{"foo", "foo"},
	{"()", "()"},
	{"foo()", "foo()"},
	{"foo()()()", "foo()()()"},
	{"foo()((()()))", "foo()((()()))"},
	{"foo()((()()))bar", "foo()((()()))bar"},
	{"foo)", "foo"},
	{"foo))", "foo"},
	{"foo)))))", "foo"},
	{"(foo", ""},
	{"((foo", ""},
	{"(((((foo", ""},
	{"(foo)", "(foo)"},
	{"((((foo))))", "((((foo))))"},
	{"foo()())", "foo()()"},
	{"foo((()())", "foo"},
	{"foo((()())) (() foo ", "foo((()())) "},
}

func TestPairedParensPrefixLen(t *testing.T) {
	for i, tt := range pairedParensPrefixLenTests {
		if out := tt.in[:pairedParensPrefixLen(tt.in)]; out != tt.out {
			t.Errorf("#%d: mismatch\nhave: %q\nwant: %q", i, out, tt.out)
		}
	}
}

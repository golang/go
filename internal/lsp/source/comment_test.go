// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"reflect"
	"strings"
	"testing"
)

// This file is a copy of go/doc/comment_test.go with the exception for
// the test cases for TestEmphasize and TestCommentEscape

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
	{
		in: "Para.\n\tshould not be ``escaped''",
		out: []block{
			{opPara, []string{"Para.\n"}},
			{opPre, []string{"should not be ``escaped''"}},
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

// This has been modified from go/doc to use markdown links instead of html ones
// and use markdown escaping instead oh html
var emphasizeTests = []struct {
	in, out string
}{
	{"", ""},
	{"http://[::1]:8080/foo.txt", `[http\:\/\/\[\:\:1\]\:8080\/foo\.txt](http://[::1]:8080/foo.txt)`},
	{"before (https://www.google.com) after", `before \([https\:\/\/www\.google\.com](https://www.google.com)\) after`},
	{"before https://www.google.com:30/x/y/z:b::c. After", `before [https\:\/\/www\.google\.com\:30\/x\/y\/z\:b\:\:c](https://www.google.com:30/x/y/z:b::c)\. After`},
	{"http://www.google.com/path/:;!-/?query=%34b#093124", `[http\:\/\/www\.google\.com\/path\/\:\;\!\-\/\?query\=\%34b\#093124](http://www.google.com/path/:;!-/?query=%34b#093124)`},
	{"http://www.google.com/path/:;!-/?query=%34bar#093124", `[http\:\/\/www\.google\.com\/path\/\:\;\!\-\/\?query\=\%34bar\#093124](http://www.google.com/path/:;!-/?query=%34bar#093124)`},
	{"http://www.google.com/index.html! After", `[http\:\/\/www\.google\.com\/index\.html](http://www.google.com/index.html)\! After`},
	{"http://www.google.com/", `[http\:\/\/www\.google\.com\/](http://www.google.com/)`},
	{"https://www.google.com/", `[https\:\/\/www\.google\.com\/](https://www.google.com/)`},
	{"http://www.google.com/path.", `[http\:\/\/www\.google\.com\/path](http://www.google.com/path)\.`},
	{"http://en.wikipedia.org/wiki/Camellia_(cipher)", `[http\:\/\/en\.wikipedia\.org\/wiki\/Camellia\_\(cipher\)](http://en.wikipedia.org/wiki/Camellia_\(cipher\))`},
	{"(http://www.google.com/)", `\([http\:\/\/www\.google\.com\/](http://www.google.com/)\)`},
	{"http://gmail.com)", `[http\:\/\/gmail\.com](http://gmail.com)\)`},
	{"((http://gmail.com))", `\(\([http\:\/\/gmail\.com](http://gmail.com)\)\)`},
	{"http://gmail.com ((http://gmail.com)) ()", `[http\:\/\/gmail\.com](http://gmail.com) \(\([http\:\/\/gmail\.com](http://gmail.com)\)\) \(\)`},
	{"Foo bar http://example.com/ quux!", `Foo bar [http\:\/\/example\.com\/](http://example.com/) quux\!`},
	{"Hello http://example.com/%2f/ /world.", `Hello [http\:\/\/example\.com\/\%2f\/](http://example.com/%2f/) \/world\.`},
	{"Lorem http: ipsum //host/path", `Lorem http\: ipsum \/\/host\/path`},
	{"javascript://is/not/linked", `javascript\:\/\/is\/not\/linked`},
	{"http://foo", `[http\:\/\/foo](http://foo)`},
	{"art by [[https://www.example.com/person/][Person Name]]", `art by \[\[[https\:\/\/www\.example\.com\/person\/](https://www.example.com/person/)\]\[Person Name\]\]`},
	{"please visit (http://golang.org/)", `please visit \([http\:\/\/golang\.org\/](http://golang.org/)\)`},
	{"please visit http://golang.org/hello())", `please visit [http\:\/\/golang\.org\/hello\(\)](http://golang.org/hello\(\))\)`},
	{"http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD", `[http\:\/\/git\.qemu\.org\/\?p\=qemu\.git\;a\=blob\;f\=qapi\-schema\.json\;hb\=HEAD](http://git.qemu.org/?p=qemu.git;a=blob;f=qapi-schema.json;hb=HEAD)`},
	{"https://foo.bar/bal/x(])", `[https\:\/\/foo\.bar\/bal\/x\(](https://foo.bar/bal/x\()\]\)`},
	{"foo [ http://bar(])", `foo \[ [http\:\/\/bar\(](http://bar\()\]\)`},
}

func TestEmphasize(t *testing.T) {
	for i, tt := range emphasizeTests {
		var buf bytes.Buffer
		emphasize(&buf, tt.in, true)
		out := buf.String()
		if out != tt.out {
			t.Errorf("#%d: mismatch\nhave: %v\nwant: %v", i, out, tt.out)
		}
	}
}

func TestCommentEscape(t *testing.T) {
	//ldquo -> ulquo and rdquo -> urquo
	commentTests := []struct {
		in, out string
	}{
		{"typically invoked as ``go tool asm'',", "typically invoked as " + ulquo + "go tool asm" + urquo + ","},
		{"For more detail, run ``go help test'' and ``go help testflag''", "For more detail, run " + ulquo + "go help test" + urquo + " and " + ulquo + "go help testflag" + urquo}}
	for i, tt := range commentTests {
		var buf strings.Builder
		commentEscape(&buf, tt.in, true)
		out := buf.String()
		if out != tt.out {
			t.Errorf("#%d: mismatch\nhave: %q\nwant: %q", i, out, tt.out)
		}
	}
}

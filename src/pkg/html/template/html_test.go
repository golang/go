// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"html"
	"strings"
	"testing"
)

func TestHTMLNospaceEscaper(t *testing.T) {
	input := ("\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f" +
		"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f" +
		` !"#$%&'()*+,-./` +
		`0123456789:;<=>?` +
		`@ABCDEFGHIJKLMNO` +
		`PQRSTUVWXYZ[\]^_` +
		"`abcdefghijklmno" +
		"pqrstuvwxyz{|}~\x7f" +
		"\u00A0\u0100\u2028\u2029\ufeff\ufdec\U0001D11E")

	want := ("&#xfffd;\x01\x02\x03\x04\x05\x06\x07" +
		"\x08&#9;&#10;&#11;&#12;&#13;\x0E\x0F" +
		"\x10\x11\x12\x13\x14\x15\x16\x17" +
		"\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f" +
		`&#32;!&#34;#$%&amp;&#39;()*&#43;,-./` +
		`0123456789:;&lt;&#61;&gt;?` +
		`@ABCDEFGHIJKLMNO` +
		`PQRSTUVWXYZ[\]^_` +
		`&#96;abcdefghijklmno` +
		`pqrstuvwxyz{|}~` + "\u007f" +
		"\u00A0\u0100\u2028\u2029\ufeff&#xfdec;\U0001D11E")

	got := htmlNospaceEscaper(input)
	if got != want {
		t.Errorf("encode: want\n\t%q\nbut got\n\t%q", want, got)
	}

	got, want = html.UnescapeString(got), strings.Replace(input, "\x00", "\ufffd", 1)
	if want != got {
		t.Errorf("decode: want\n\t%q\nbut got\n\t%q", want, got)
	}
}

func TestStripTags(t *testing.T) {
	tests := []struct {
		input, want string
	}{
		{"", ""},
		{"Hello, World!", "Hello, World!"},
		{"foo&amp;bar", "foo&amp;bar"},
		{`Hello <a href="www.example.com/">World</a>!`, "Hello World!"},
		{"Foo <textarea>Bar</textarea> Baz", "Foo Bar Baz"},
		{"Foo <!-- Bar --> Baz", "Foo  Baz"},
		{"<", "<"},
		{"foo < bar", "foo < bar"},
		{`Foo<script type="text/javascript">alert(1337)</script>Bar`, "FooBar"},
		{`Foo<div title="1>2">Bar`, "FooBar"},
		{`I <3 Ponies!`, `I <3 Ponies!`},
		{`<script>foo()</script>`, ``},
	}

	for _, test := range tests {
		if got := stripTags(test.input); got != test.want {
			t.Errorf("%q: want %q, got %q", test.input, test.want, got)
		}
	}
}

func BenchmarkHTMLNospaceEscaper(b *testing.B) {
	for i := 0; i < b.N; i++ {
		htmlNospaceEscaper("The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>")
	}
}

func BenchmarkHTMLNospaceEscaperNoSpecials(b *testing.B) {
	for i := 0; i < b.N; i++ {
		htmlNospaceEscaper("The_quick,_brown_fox_jumps_over_the_lazy_dog.")
	}
}

func BenchmarkStripTags(b *testing.B) {
	for i := 0; i < b.N; i++ {
		stripTags("The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>")
	}
}

func BenchmarkStripTagsNoSpecials(b *testing.B) {
	for i := 0; i < b.N; i++ {
		stripTags("The quick, brown fox jumps over the lazy dog.")
	}
}

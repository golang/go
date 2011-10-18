// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package html

import (
	"strconv"
	"strings"
	"testing"
)

func TestEndsWithCSSKeyword(t *testing.T) {
	tests := []struct {
		css, kw string
		want    bool
	}{
		{"", "url", false},
		{"url", "url", true},
		{"URL", "url", true},
		{"Url", "url", true},
		{"url", "important", false},
		{"important", "important", true},
		{"image-url", "url", false},
		{"imageurl", "url", false},
		{"image url", "url", true},
	}
	for _, test := range tests {
		got := endsWithCSSKeyword([]byte(test.css), test.kw)
		if got != test.want {
			t.Errorf("want %t but got %t for css=%v, kw=%v", test.want, got, test.css, test.kw)
		}
	}
}

func TestIsCSSNmchar(t *testing.T) {
	tests := []struct {
		rune int
		want bool
	}{
		{0, false},
		{'0', true},
		{'9', true},
		{'A', true},
		{'Z', true},
		{'a', true},
		{'z', true},
		{'_', true},
		{'-', true},
		{':', false},
		{';', false},
		{' ', false},
		{0x7f, false},
		{0x80, true},
		{0x1234, true},
		{0xd800, false},
		{0xdc00, false},
		{0xfffe, false},
		{0x10000, true},
		{0x110000, false},
	}
	for _, test := range tests {
		got := isCSSNmchar(test.rune)
		if got != test.want {
			t.Errorf("%q: want %t but got %t", string(test.rune), test.want, got)
		}
	}
}

func TestDecodeCSS(t *testing.T) {
	tests := []struct {
		css, want string
	}{
		{``, ``},
		{`foo`, `foo`},
		{`foo\`, `foo`},
		{`foo\\`, `foo\`},
		{`\`, ``},
		{`\A`, "\n"},
		{`\a`, "\n"},
		{`\0a`, "\n"},
		{`\00000a`, "\n"},
		{`\000000a`, "\u0000a"},
		{`\1234 5`, "\u1234" + "5"},
		{`\1234\20 5`, "\u1234" + " 5"},
		{`\1234\A 5`, "\u1234" + "\n5"},
		{"\\1234\t5", "\u1234" + "5"},
		{"\\1234\n5", "\u1234" + "5"},
		{"\\1234\r\n5", "\u1234" + "5"},
		{`\12345`, "\U00012345"},
		{`\\`, `\`},
		{`\\ `, `\ `},
		{`\"`, `"`},
		{`\'`, `'`},
		{`\.`, `.`},
		{`\. .`, `. .`},
		{
			`The \3c i\3equick\3c/i\3e,\d\A\3cspan style=\27 color:brown\27\3e brown\3c/span\3e  fox jumps\2028over the \3c canine class=\22lazy\22 \3e dog\3c/canine\3e`,
			"The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>",
		},
	}
	for _, test := range tests {
		got1 := string(decodeCSS([]byte(test.css)))
		if got1 != test.want {
			t.Errorf("%q: want\n\t%q\nbut got\n\t%q", test.css, test.want, got1)
		}
		recoded := cssEscaper(got1)
		if got2 := string(decodeCSS([]byte(recoded))); got2 != test.want {
			t.Errorf("%q: escape & decode not dual for %q", test.css, recoded)
		}
	}
}

func TestHexDecode(t *testing.T) {
	for i := 0; i < 0x200000; i += 101 /* coprime with 16 */ {
		s := strconv.Itob(i, 16)
		if got := hexDecode([]byte(s)); got != i {
			t.Errorf("%s: want %d but got %d", s, i, got)
		}
		s = strings.ToUpper(s)
		if got := hexDecode([]byte(s)); got != i {
			t.Errorf("%s: want %d but got %d", s, i, got)
		}
	}
}

func TestSkipCSSSpace(t *testing.T) {
	tests := []struct {
		css, want string
	}{
		{"", ""},
		{"foo", "foo"},
		{"\n", ""},
		{"\r\n", ""},
		{"\r", ""},
		{"\t", ""},
		{" ", ""},
		{"\f", ""},
		{" foo", "foo"},
		{"  foo", " foo"},
		{`\20`, `\20`},
	}
	for _, test := range tests {
		got := string(skipCSSSpace([]byte(test.css)))
		if got != test.want {
			t.Errorf("%q: want %q but got %q", test.css, test.want, got)
		}
	}
}

func TestCSSEscaper(t *testing.T) {
	input := ("\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f" +
		"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f" +
		` !"#$%&'()*+,-./` +
		`0123456789:;<=>?` +
		`@ABCDEFGHIJKLMNO` +
		`PQRSTUVWXYZ[\]^_` +
		"`abcdefghijklmno" +
		"pqrstuvwxyz{|}~\x7f" +
		"\u00A0\u0100\u2028\u2029\ufeff\U0001D11E")

	want := ("\\0\x01\x02\x03\x04\x05\x06\x07" +
		"\x08\\9 \\a\x0b\\c \\d\x0E\x0F" +
		"\x10\x11\x12\x13\x14\x15\x16\x17" +
		"\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f" +
		` !\22#$%\26\27\28\29*\2b,-.\2f ` +
		`0123456789\3a\3b\3c=\3e?` +
		`@ABCDEFGHIJKLMNO` +
		`PQRSTUVWXYZ[\\]^_` +
		"`abcdefghijklmno" +
		`pqrstuvwxyz\7b|\7d~` + "\u007f" +
		"\u00A0\u0100\u2028\u2029\ufeff\U0001D11E")

	got := cssEscaper(input)
	if got != want {
		t.Errorf("encode: want\n\t%q\nbut got\n\t%q", want, got)
	}

	got = string(decodeCSS([]byte(got)))
	if input != got {
		t.Errorf("decode: want\n\t%q\nbut got\n\t%q", input, got)
	}
}

func TestCSSValueFilter(t *testing.T) {
	tests := []struct {
		css, want string
	}{
		{"", ""},
		{"foo", "foo"},
		{"0", "0"},
		{"0px", "0px"},
		{"-5px", "-5px"},
		{"1.25in", "1.25in"},
		{"+.33em", "+.33em"},
		{"100%", "100%"},
		{"12.5%", "12.5%"},
		{".foo", ".foo"},
		{"#bar", "#bar"},
		{"corner-radius", "corner-radius"},
		{"-moz-corner-radius", "-moz-corner-radius"},
		{"#000", "#000"},
		{"#48f", "#48f"},
		{"#123456", "#123456"},
		{"U+00-FF, U+980-9FF", "U+00-FF, U+980-9FF"},
		{"color: red", "color: red"},
		{"<!--", "ZgotmplZ"},
		{"-->", "ZgotmplZ"},
		{"<![CDATA[", "ZgotmplZ"},
		{"]]>", "ZgotmplZ"},
		{"</style", "ZgotmplZ"},
		{`"`, "ZgotmplZ"},
		{`'`, "ZgotmplZ"},
		{"`", "ZgotmplZ"},
		{"\x00", "ZgotmplZ"},
		{"/* foo */", "ZgotmplZ"},
		{"//", "ZgotmplZ"},
		{"[href=~", "ZgotmplZ"},
		{"expression(alert(1337))", "ZgotmplZ"},
		{"-expression(alert(1337))", "ZgotmplZ"},
		{"expression", "ZgotmplZ"},
		{"Expression", "ZgotmplZ"},
		{"EXPRESSION", "ZgotmplZ"},
		{"-moz-binding", "ZgotmplZ"},
		{"-expr\x00ession(alert(1337))", "ZgotmplZ"},
		{`-expr\0ession(alert(1337))`, "ZgotmplZ"},
		{`-express\69on(alert(1337))`, "ZgotmplZ"},
		{`-express\69 on(alert(1337))`, "ZgotmplZ"},
		{`-exp\72 ession(alert(1337))`, "ZgotmplZ"},
		{`-exp\52 ession(alert(1337))`, "ZgotmplZ"},
		{`-exp\000052 ession(alert(1337))`, "ZgotmplZ"},
		{`-expre\0000073sion`, "-expre\x073sion"},
		{`@import url evil.css`, "ZgotmplZ"},
	}
	for _, test := range tests {
		got := cssValueFilter(test.css)
		if got != test.want {
			t.Errorf("%q: want %q but got %q", test.css, test.want, got)
		}
	}
}

func BenchmarkCSSEscaper(b *testing.B) {
	for i := 0; i < b.N; i++ {
		cssEscaper("The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>")
	}
}

func BenchmarkCSSEscaperNoSpecials(b *testing.B) {
	for i := 0; i < b.N; i++ {
		cssEscaper("The quick, brown fox jumps over the lazy dog.")
	}
}

func BenchmarkDecodeCSS(b *testing.B) {
	s := []byte(`The \3c i\3equick\3c/i\3e,\d\A\3cspan style=\27 color:brown\27\3e brown\3c/span\3e fox jumps\2028over the \3c canine class=\22lazy\22 \3edog\3c/canine\3e`)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeCSS(s)
	}
}

func BenchmarkDecodeCSSNoSpecials(b *testing.B) {
	s := []byte("The quick, brown fox jumps over the lazy dog.")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		decodeCSS(s)
	}
}

func BenchmarkCSSValueFilter(b *testing.B) {
	for i := 0; i < b.N; i++ {
		cssValueFilter(`  e\78preS\0Sio/**/n(alert(1337))`)
	}
}

func BenchmarkCSSValueFilterOk(b *testing.B) {
	for i := 0; i < b.N; i++ {
		cssValueFilter(`Times New Roman`)
	}
}

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package template

import (
	"errors"
	"math"
	"strings"
	"testing"
)

func TestNextJsCtx(t *testing.T) {
	tests := []struct {
		jsCtx jsCtx
		s     string
	}{
		// Statement terminators precede regexps.
		{jsCtxRegexp, ";"},
		// This is not airtight.
		//     ({ valueOf: function () { return 1 } } / 2)
		// is valid JavaScript but in practice, devs do not do this.
		// A block followed by a statement starting with a RegExp is
		// much more common:
		//     while (x) {...} /foo/.test(x) || panic()
		{jsCtxRegexp, "}"},
		// But member, call, grouping, and array expression terminators
		// precede div ops.
		{jsCtxDivOp, ")"},
		{jsCtxDivOp, "]"},
		// At the start of a primary expression, array, or expression
		// statement, expect a regexp.
		{jsCtxRegexp, "("},
		{jsCtxRegexp, "["},
		{jsCtxRegexp, "{"},
		// Assignment operators precede regexps as do all exclusively
		// prefix and binary operators.
		{jsCtxRegexp, "="},
		{jsCtxRegexp, "+="},
		{jsCtxRegexp, "*="},
		{jsCtxRegexp, "*"},
		{jsCtxRegexp, "!"},
		// Whether the + or - is infix or prefix, it cannot precede a
		// div op.
		{jsCtxRegexp, "+"},
		{jsCtxRegexp, "-"},
		// An incr/decr op precedes a div operator.
		// This is not airtight. In (g = ++/h/i) a regexp follows a
		// pre-increment operator, but in practice devs do not try to
		// increment or decrement regular expressions.
		// (g++/h/i) where ++ is a postfix operator on g is much more
		// common.
		{jsCtxDivOp, "--"},
		{jsCtxDivOp, "++"},
		{jsCtxDivOp, "x--"},
		// When we have many dashes or pluses, then they are grouped
		// left to right.
		{jsCtxRegexp, "x---"}, // A postfix -- then a -.
		// return followed by a slash returns the regexp literal or the
		// slash starts a regexp literal in an expression statement that
		// is dead code.
		{jsCtxRegexp, "return"},
		{jsCtxRegexp, "return "},
		{jsCtxRegexp, "return\t"},
		{jsCtxRegexp, "return\n"},
		{jsCtxRegexp, "return\u2028"},
		// Identifiers can be divided and cannot validly be preceded by
		// a regular expressions. Semicolon insertion cannot happen
		// between an identifier and a regular expression on a new line
		// because the one token lookahead for semicolon insertion has
		// to conclude that it could be a div binary op and treat it as
		// such.
		{jsCtxDivOp, "x"},
		{jsCtxDivOp, "x "},
		{jsCtxDivOp, "x\t"},
		{jsCtxDivOp, "x\n"},
		{jsCtxDivOp, "x\u2028"},
		{jsCtxDivOp, "preturn"},
		// Numbers precede div ops.
		{jsCtxDivOp, "0"},
		// Dots that are part of a number are div preceders.
		{jsCtxDivOp, "0."},
		// Some JS interpreters treat NBSP as a normal space, so
		// we must too in order to properly escape things.
		{jsCtxRegexp, "=\u00A0"},
	}

	for _, test := range tests {
		if ctx := nextJSCtx([]byte(test.s), jsCtxRegexp); ctx != test.jsCtx {
			t.Errorf("%q: want %s got %s", test.s, test.jsCtx, ctx)
		}
		if ctx := nextJSCtx([]byte(test.s), jsCtxDivOp); ctx != test.jsCtx {
			t.Errorf("%q: want %s got %s", test.s, test.jsCtx, ctx)
		}
	}

	if nextJSCtx([]byte("   "), jsCtxRegexp) != jsCtxRegexp {
		t.Error("Blank tokens")
	}

	if nextJSCtx([]byte("   "), jsCtxDivOp) != jsCtxDivOp {
		t.Error("Blank tokens")
	}
}

type jsonErrType struct{}

func (e *jsonErrType) MarshalJSON() ([]byte, error) {
	return nil, errors.New("a */ b <script c </script d <!-- e <sCrIpT f </sCrIpT")
}

func TestJSValEscaper(t *testing.T) {
	tests := []struct {
		x        any
		js       string
		skipNest bool
	}{
		{int(42), " 42 ", false},
		{uint(42), " 42 ", false},
		{int16(42), " 42 ", false},
		{uint16(42), " 42 ", false},
		{int32(-42), " -42 ", false},
		{uint32(42), " 42 ", false},
		{int16(-42), " -42 ", false},
		{uint16(42), " 42 ", false},
		{int64(-42), " -42 ", false},
		{uint64(42), " 42 ", false},
		{uint64(1) << 53, " 9007199254740992 ", false},
		// ulp(1 << 53) > 1 so this loses precision in JS
		// but it is still a representable integer literal.
		{uint64(1)<<53 + 1, " 9007199254740993 ", false},
		{float32(1.0), " 1 ", false},
		{float32(-1.0), " -1 ", false},
		{float32(0.5), " 0.5 ", false},
		{float32(-0.5), " -0.5 ", false},
		{float32(1.0) / float32(256), " 0.00390625 ", false},
		{float32(0), " 0 ", false},
		{math.Copysign(0, -1), " -0 ", false},
		{float64(1.0), " 1 ", false},
		{float64(-1.0), " -1 ", false},
		{float64(0.5), " 0.5 ", false},
		{float64(-0.5), " -0.5 ", false},
		{float64(0), " 0 ", false},
		{math.Copysign(0, -1), " -0 ", false},
		{"", `""`, false},
		{"foo", `"foo"`, false},
		// Newlines.
		{"\r\n\u2028\u2029", `"\r\n\u2028\u2029"`, false},
		// "\v" == "v" on IE 6 so use "\u000b" instead.
		{"\t\x0b", `"\t\u000b"`, false},
		{struct{ X, Y int }{1, 2}, `{"X":1,"Y":2}`, false},
		{[]any{}, "[]", false},
		{[]any{42, "foo", nil}, `[42,"foo",null]`, false},
		{[]string{"<!--", "</script>", "-->"}, `["\u003c!--","\u003c/script\u003e","--\u003e"]`, false},
		{"<!--", `"\u003c!--"`, false},
		{"-->", `"--\u003e"`, false},
		{"<![CDATA[", `"\u003c![CDATA["`, false},
		{"]]>", `"]]\u003e"`, false},
		{"</script", `"\u003c/script"`, false},
		{"\U0001D11E", "\"\U0001D11E\"", false}, // or "\uD834\uDD1E"
		{nil, " null ", false},
		{&jsonErrType{}, " /* json: error calling MarshalJSON for type *template.jsonErrType: a * / b \\x3Cscript c \\x3C/script d \\x3C!-- e \\x3Cscript f \\x3C/script */null ", true},
	}

	for _, test := range tests {
		if js := jsValEscaper(test.x); js != test.js {
			t.Errorf("%+v: want\n\t%q\ngot\n\t%q", test.x, test.js, js)
		}
		if test.skipNest {
			continue
		}
		// Make sure that escaping corner cases are not broken
		// by nesting.
		a := []any{test.x}
		want := "[" + strings.TrimSpace(test.js) + "]"
		if js := jsValEscaper(a); js != want {
			t.Errorf("%+v: want\n\t%q\ngot\n\t%q", a, want, js)
		}
	}
}

func TestJSStrEscaper(t *testing.T) {
	tests := []struct {
		x   any
		esc string
	}{
		{"", ``},
		{"foo", `foo`},
		{"\u0000", `\u0000`},
		{"\t", `\t`},
		{"\n", `\n`},
		{"\r", `\r`},
		{"\u2028", `\u2028`},
		{"\u2029", `\u2029`},
		{"\\", `\\`},
		{"\\n", `\\n`},
		{"foo\r\nbar", `foo\r\nbar`},
		// Preserve attribute boundaries.
		{`"`, `\u0022`},
		{`'`, `\u0027`},
		// Allow embedding in HTML without further escaping.
		{`&amp;`, `\u0026amp;`},
		// Prevent breaking out of text node and element boundaries.
		{"</script>", `\u003c\/script\u003e`},
		{"<![CDATA[", `\u003c![CDATA[`},
		{"]]>", `]]\u003e`},
		// https://dev.w3.org/html5/markup/aria/syntax.html#escaping-text-span
		//   "The text in style, script, title, and textarea elements
		//   must not have an escaping text span start that is not
		//   followed by an escaping text span end."
		// Furthermore, spoofing an escaping text span end could lead
		// to different interpretation of a </script> sequence otherwise
		// masked by the escaping text span, and spoofing a start could
		// allow regular text content to be interpreted as script
		// allowing script execution via a combination of a JS string
		// injection followed by an HTML text injection.
		{"<!--", `\u003c!--`},
		{"-->", `--\u003e`},
		// From https://code.google.com/p/doctype/wiki/ArticleUtf7
		{"+ADw-script+AD4-alert(1)+ADw-/script+AD4-",
			`\u002bADw-script\u002bAD4-alert(1)\u002bADw-\/script\u002bAD4-`,
		},
		// Invalid UTF-8 sequence
		{"foo\xA0bar", "foo\xA0bar"},
		// Invalid unicode scalar value.
		{"foo\xed\xa0\x80bar", "foo\xed\xa0\x80bar"},
	}

	for _, test := range tests {
		esc := jsStrEscaper(test.x)
		if esc != test.esc {
			t.Errorf("%q: want %q got %q", test.x, test.esc, esc)
		}
	}
}

func TestJSRegexpEscaper(t *testing.T) {
	tests := []struct {
		x   any
		esc string
	}{
		{"", `(?:)`},
		{"foo", `foo`},
		{"\u0000", `\u0000`},
		{"\t", `\t`},
		{"\n", `\n`},
		{"\r", `\r`},
		{"\u2028", `\u2028`},
		{"\u2029", `\u2029`},
		{"\\", `\\`},
		{"\\n", `\\n`},
		{"foo\r\nbar", `foo\r\nbar`},
		// Preserve attribute boundaries.
		{`"`, `\u0022`},
		{`'`, `\u0027`},
		// Allow embedding in HTML without further escaping.
		{`&amp;`, `\u0026amp;`},
		// Prevent breaking out of text node and element boundaries.
		{"</script>", `\u003c\/script\u003e`},
		{"<![CDATA[", `\u003c!\[CDATA\[`},
		{"]]>", `\]\]\u003e`},
		// Escaping text spans.
		{"<!--", `\u003c!\-\-`},
		{"-->", `\-\-\u003e`},
		{"*", `\*`},
		{"+", `\u002b`},
		{"?", `\?`},
		{"[](){}", `\[\]\(\)\{\}`},
		{"$foo|x.y", `\$foo\|x\.y`},
		{"x^y", `x\^y`},
	}

	for _, test := range tests {
		esc := jsRegexpEscaper(test.x)
		if esc != test.esc {
			t.Errorf("%q: want %q got %q", test.x, test.esc, esc)
		}
	}
}

func TestEscapersOnLower7AndSelectHighCodepoints(t *testing.T) {
	input := ("\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f" +
		"\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f" +
		` !"#$%&'()*+,-./` +
		`0123456789:;<=>?` +
		`@ABCDEFGHIJKLMNO` +
		`PQRSTUVWXYZ[\]^_` +
		"`abcdefghijklmno" +
		"pqrstuvwxyz{|}~\x7f" +
		"\u00A0\u0100\u2028\u2029\ufeff\U0001D11E")

	tests := []struct {
		name    string
		escaper func(...any) string
		escaped string
	}{
		{
			"jsStrEscaper",
			jsStrEscaper,
			`\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007` +
				`\u0008\t\n\u000b\f\r\u000e\u000f` +
				`\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017` +
				`\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f` +
				` !\u0022#$%\u0026\u0027()*\u002b,-.\/` +
				`0123456789:;\u003c=\u003e?` +
				`@ABCDEFGHIJKLMNO` +
				`PQRSTUVWXYZ[\\]^_` +
				"\\u0060abcdefghijklmno" +
				"pqrstuvwxyz{|}~\u007f" +
				"\u00A0\u0100\\u2028\\u2029\ufeff\U0001D11E",
		},
		{
			"jsRegexpEscaper",
			jsRegexpEscaper,
			`\u0000\u0001\u0002\u0003\u0004\u0005\u0006\u0007` +
				`\u0008\t\n\u000b\f\r\u000e\u000f` +
				`\u0010\u0011\u0012\u0013\u0014\u0015\u0016\u0017` +
				`\u0018\u0019\u001a\u001b\u001c\u001d\u001e\u001f` +
				` !\u0022#\$%\u0026\u0027\(\)\*\u002b,\-\.\/` +
				`0123456789:;\u003c=\u003e\?` +
				`@ABCDEFGHIJKLMNO` +
				`PQRSTUVWXYZ\[\\\]\^_` +
				"`abcdefghijklmno" +
				`pqrstuvwxyz\{\|\}~` + "\u007f" +
				"\u00A0\u0100\\u2028\\u2029\ufeff\U0001D11E",
		},
	}

	for _, test := range tests {
		if s := test.escaper(input); s != test.escaped {
			t.Errorf("%s once: want\n\t%q\ngot\n\t%q", test.name, test.escaped, s)
			continue
		}

		// Escape it rune by rune to make sure that any
		// fast-path checking does not break escaping.
		var buf strings.Builder
		for _, c := range input {
			buf.WriteString(test.escaper(string(c)))
		}

		if s := buf.String(); s != test.escaped {
			t.Errorf("%s rune-wise: want\n\t%q\ngot\n\t%q", test.name, test.escaped, s)
			continue
		}
	}
}

func TestIsJsMimeType(t *testing.T) {
	tests := []struct {
		in  string
		out bool
	}{
		{"application/javascript;version=1.8", true},
		{"application/javascript;version=1.8;foo=bar", true},
		{"application/javascript/version=1.8", false},
		{"text/javascript", true},
		{"application/json", true},
		{"application/ld+json", true},
		{"module", true},
	}

	for _, test := range tests {
		if isJSType(test.in) != test.out {
			t.Errorf("isJSType(%q) = %v, want %v", test.in, !test.out, test.out)
		}
	}
}

func BenchmarkJSValEscaperWithNum(b *testing.B) {
	for i := 0; i < b.N; i++ {
		jsValEscaper(3.141592654)
	}
}

func BenchmarkJSValEscaperWithStr(b *testing.B) {
	for i := 0; i < b.N; i++ {
		jsValEscaper("The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>")
	}
}

func BenchmarkJSValEscaperWithStrNoSpecials(b *testing.B) {
	for i := 0; i < b.N; i++ {
		jsValEscaper("The quick, brown fox jumps over the lazy dog")
	}
}

func BenchmarkJSValEscaperWithObj(b *testing.B) {
	o := struct {
		S string
		N int
	}{
		"The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>\u2028",
		42,
	}
	for i := 0; i < b.N; i++ {
		jsValEscaper(o)
	}
}

func BenchmarkJSValEscaperWithObjNoSpecials(b *testing.B) {
	o := struct {
		S string
		N int
	}{
		"The quick, brown fox jumps over the lazy dog",
		42,
	}
	for i := 0; i < b.N; i++ {
		jsValEscaper(o)
	}
}

func BenchmarkJSStrEscaperNoSpecials(b *testing.B) {
	for i := 0; i < b.N; i++ {
		jsStrEscaper("The quick, brown fox jumps over the lazy dog.")
	}
}

func BenchmarkJSStrEscaper(b *testing.B) {
	for i := 0; i < b.N; i++ {
		jsStrEscaper("The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>")
	}
}

func BenchmarkJSRegexpEscaperNoSpecials(b *testing.B) {
	for i := 0; i < b.N; i++ {
		jsRegexpEscaper("The quick, brown fox jumps over the lazy dog")
	}
}

func BenchmarkJSRegexpEscaper(b *testing.B) {
	for i := 0; i < b.N; i++ {
		jsRegexpEscaper("The <i>quick</i>,\r\n<span style='color:brown'>brown</span> fox jumps\u2028over the <canine class=\"lazy\">dog</canine>")
	}
}

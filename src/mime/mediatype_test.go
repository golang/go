// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"reflect"
	"testing"
)

func TestConsumeToken(t *testing.T) {
	tests := [...][3]string{
		{"foo bar", "foo", " bar"},
		{"bar", "bar", ""},
		{"", "", ""},
		{" foo", "", " foo"},
	}
	for _, test := range tests {
		token, rest := consumeToken(test[0])
		expectedToken := test[1]
		expectedRest := test[2]
		if token != expectedToken {
			t.Errorf("expected to consume token '%s', not '%s' from '%s'",
				expectedToken, token, test[0])
		} else if rest != expectedRest {
			t.Errorf("expected to have left '%s', not '%s' after reading token '%s' from '%s'",
				expectedRest, rest, token, test[0])
		}
	}
}

func TestConsumeValue(t *testing.T) {
	tests := [...][3]string{
		{"foo bar", "foo", " bar"},
		{"bar", "bar", ""},
		{" bar ", "", " bar "},
		{`"My value"end`, "My value", "end"},
		{`"My value" end`, "My value", " end"},
		{`"\\" rest`, "\\", " rest"},
		{`"My \" value"end`, "My \" value", "end"},
		{`"\" rest`, "", `"\" rest`},
	}
	for _, test := range tests {
		value, rest := consumeValue(test[0])
		expectedValue := test[1]
		expectedRest := test[2]
		if value != expectedValue {
			t.Errorf("expected to consume value [%s], not [%s] from [%s]",
				expectedValue, value, test[0])
		} else if rest != expectedRest {
			t.Errorf("expected to have left [%s], not [%s] after reading value [%s] from [%s]",
				expectedRest, rest, value, test[0])
		}
	}
}

func TestConsumeMediaParam(t *testing.T) {
	tests := [...][4]string{
		{" ; foo=bar", "foo", "bar", ""},
		{"; foo=bar", "foo", "bar", ""},
		{";foo=bar", "foo", "bar", ""},
		{";FOO=bar", "foo", "bar", ""},
		{`;foo="bar"`, "foo", "bar", ""},
		{`;foo="bar"; `, "foo", "bar", "; "},
		{`;foo="bar"; foo=baz`, "foo", "bar", "; foo=baz"},
		{` ; boundary=----CUT;`, "boundary", "----CUT", ";"},
		{` ; key=value;  blah="value";name="foo" `, "key", "value", `;  blah="value";name="foo" `},
		{`;  blah="value";name="foo" `, "blah", "value", `;name="foo" `},
		{`;name="foo" `, "name", "foo", ` `},
	}
	for _, test := range tests {
		param, value, rest := consumeMediaParam(test[0])
		expectedParam := test[1]
		expectedValue := test[2]
		expectedRest := test[3]
		if param != expectedParam {
			t.Errorf("expected to consume param [%s], not [%s] from [%s]",
				expectedParam, param, test[0])
		} else if value != expectedValue {
			t.Errorf("expected to consume value [%s], not [%s] from [%s]",
				expectedValue, value, test[0])
		} else if rest != expectedRest {
			t.Errorf("expected to have left [%s], not [%s] after reading [%s/%s] from [%s]",
				expectedRest, rest, param, value, test[0])
		}
	}
}

type mediaTypeTest struct {
	in string
	t  string
	p  map[string]string
}

func TestParseMediaType(t *testing.T) {
	// Convenience map initializer
	m := func(s ...string) map[string]string {
		sm := make(map[string]string)
		for i := 0; i < len(s); i += 2 {
			sm[s[i]] = s[i+1]
		}
		return sm
	}

	nameFoo := map[string]string{"name": "foo"}
	tests := []mediaTypeTest{
		{`form-data; name="foo"`, "form-data", nameFoo},
		{` form-data ; name=foo`, "form-data", nameFoo},
		{`FORM-DATA;name="foo"`, "form-data", nameFoo},
		{` FORM-DATA ; name="foo"`, "form-data", nameFoo},
		{` FORM-DATA ; name="foo"`, "form-data", nameFoo},

		{`form-data; key=value;  blah="value";name="foo" `,
			"form-data",
			m("key", "value", "blah", "value", "name", "foo")},

		{`foo; key=val1; key=the-key-appears-again-which-is-bogus`,
			"", m()},

		// From RFC 2231:
		{`application/x-stuff; title*=us-ascii'en-us'This%20is%20%2A%2A%2Afun%2A%2A%2A`,
			"application/x-stuff",
			m("title", "This is ***fun***")},

		{`message/external-body; access-type=URL; ` +
			`URL*0="ftp://";` +
			`URL*1="cs.utk.edu/pub/moore/bulk-mailer/bulk-mailer.tar"`,
			"message/external-body",
			m("access-type", "URL",
				"url", "ftp://cs.utk.edu/pub/moore/bulk-mailer/bulk-mailer.tar")},

		{`application/x-stuff; ` +
			`title*0*=us-ascii'en'This%20is%20even%20more%20; ` +
			`title*1*=%2A%2A%2Afun%2A%2A%2A%20; ` +
			`title*2="isn't it!"`,
			"application/x-stuff",
			m("title", "This is even more ***fun*** isn't it!")},

		// Tests from http://greenbytes.de/tech/tc2231/
		// Note: Backslash escape handling is a bit loose, like MSIE.
		// TODO(bradfitz): add the rest of the tests from that site.
		{`attachment; filename="f\oo.html"`,
			"attachment",
			m("filename", "f\\oo.html")},
		{`attachment; filename="\"quoting\" tested.html"`,
			"attachment",
			m("filename", `"quoting" tested.html`)},
		{`attachment; filename="Here's a semicolon;.html"`,
			"attachment",
			m("filename", "Here's a semicolon;.html")},
		{`attachment; foo="\"\\";filename="foo.html"`,
			"attachment",
			m("foo", "\"\\", "filename", "foo.html")},
		{`attachment; filename=foo.html`,
			"attachment",
			m("filename", "foo.html")},
		{`attachment; filename=foo.html ;`,
			"attachment",
			m("filename", "foo.html")},
		{`attachment; filename='foo.html'`,
			"attachment",
			m("filename", "'foo.html'")},
		{`attachment; filename="foo-%41.html"`,
			"attachment",
			m("filename", "foo-%41.html")},
		{`attachment; filename="foo-%\41.html"`,
			"attachment",
			m("filename", "foo-%\\41.html")},
		{`filename=foo.html`,
			"", m()},
		{`x=y; filename=foo.html`,
			"", m()},
		{`"foo; filename=bar;baz"; filename=qux`,
			"", m()},
		{`inline; attachment; filename=foo.html`,
			"", m()},
		{`attachment; filename="foo.html".txt`,
			"", m()},
		{`attachment; filename="bar`,
			"", m()},
		{`attachment; creation-date="Wed, 12 Feb 1997 16:29:51 -0500"`,
			"attachment",
			m("creation-date", "Wed, 12 Feb 1997 16:29:51 -0500")},
		{`foobar`, "foobar", m()},
		{`attachment; filename* =UTF-8''foo-%c3%a4.html`,
			"attachment",
			m("filename", "foo-ä.html")},
		{`attachment; filename*=UTF-8''A-%2541.html`,
			"attachment",
			m("filename", "A-%41.html")},
		{`attachment; filename*0="foo."; filename*1="html"`,
			"attachment",
			m("filename", "foo.html")},
		{`attachment; filename*0*=UTF-8''foo-%c3%a4; filename*1=".html"`,
			"attachment",
			m("filename", "foo-ä.html")},
		{`attachment; filename*0="foo"; filename*01="bar"`,
			"attachment",
			m("filename", "foo")},
		{`attachment; filename*0="foo"; filename*2="bar"`,
			"attachment",
			m("filename", "foo")},
		{`attachment; filename*1="foo"; filename*2="bar"`,
			"attachment", m()},
		{`attachment; filename*1="bar"; filename*0="foo"`,
			"attachment",
			m("filename", "foobar")},
		{`attachment; filename="foo-ae.html"; filename*=UTF-8''foo-%c3%a4.html`,
			"attachment",
			m("filename", "foo-ä.html")},
		{`attachment; filename*=UTF-8''foo-%c3%a4.html; filename="foo-ae.html"`,
			"attachment",
			m("filename", "foo-ä.html")},

		// Browsers also just send UTF-8 directly without RFC 2231,
		// at least when the source page is served with UTF-8.
		{`form-data; firstname="Брэд"; lastname="Фицпатрик"`,
			"form-data",
			m("firstname", "Брэд", "lastname", "Фицпатрик")},

		// Empty string used to be mishandled.
		{`foo; bar=""`, "foo", m("bar", "")},

		// Microsoft browers in intranet mode do not think they need to escape \ in file name.
		{`form-data; name="file"; filename="C:\dev\go\robots.txt"`, "form-data", m("name", "file", "filename", `C:\dev\go\robots.txt`)},
	}
	for _, test := range tests {
		mt, params, err := ParseMediaType(test.in)
		if err != nil {
			if test.t != "" {
				t.Errorf("for input %#q, unexpected error: %v", test.in, err)
				continue
			}
			continue
		}
		if g, e := mt, test.t; g != e {
			t.Errorf("for input %#q, expected type %q, got %q",
				test.in, e, g)
			continue
		}
		if len(params) == 0 && len(test.p) == 0 {
			continue
		}
		if !reflect.DeepEqual(params, test.p) {
			t.Errorf("for input %#q, wrong params.\n"+
				"expected: %#v\n"+
				"     got: %#v",
				test.in, test.p, params)
		}
	}
}

type badMediaTypeTest struct {
	in  string
	mt  string
	err string
}

var badMediaTypeTests = []badMediaTypeTest{
	{"bogus ;=========", "bogus", "mime: invalid media parameter"},
	// The following example is from real email delivered by gmail (error: missing semicolon)
	// and it is there to check behavior described in #19498
	{"application/pdf; x-mac-type=\"3F3F3F3F\"; x-mac-creator=\"3F3F3F3F\" name=\"a.pdf\";",
		"application/pdf", "mime: invalid media parameter"},
	{"bogus/<script>alert</script>", "", "mime: expected token after slash"},
	{"bogus/bogus<script>alert</script>", "", "mime: unexpected content after media subtype"},
}

func TestParseMediaTypeBogus(t *testing.T) {
	for _, tt := range badMediaTypeTests {
		mt, params, err := ParseMediaType(tt.in)
		if err == nil {
			t.Errorf("ParseMediaType(%q) = nil error; want parse error", tt.in)
			continue
		}
		if err.Error() != tt.err {
			t.Errorf("ParseMediaType(%q) = err %q; want %q", tt.in, err.Error(), tt.err)
		}
		if params != nil {
			t.Errorf("ParseMediaType(%q): got non-nil params on error", tt.in)
		}
		if err != ErrInvalidMediaParameter && mt != "" {
			t.Errorf("ParseMediaType(%q): got unexpected non-empty media type string", tt.in)
		}
		if err == ErrInvalidMediaParameter && mt != tt.mt {
			t.Errorf("ParseMediaType(%q): in case of invalid parameters: expected type %q, got %q", tt.in, tt.mt, mt)
		}
	}
}

type formatTest struct {
	typ    string
	params map[string]string
	want   string
}

var formatTests = []formatTest{
	{"noslash", map[string]string{"X": "Y"}, "noslash; x=Y"}, // e.g. Content-Disposition values (RFC 2183); issue 11289
	{"foo bar/baz", nil, ""},
	{"foo/bar baz", nil, ""},
	{"foo/BAR", nil, "foo/bar"},
	{"foo/BAR", map[string]string{"X": "Y"}, "foo/bar; x=Y"},
	{"foo/BAR", map[string]string{"space": "With space"}, `foo/bar; space="With space"`},
	{"foo/BAR", map[string]string{"quote": `With "quote`}, `foo/bar; quote="With \"quote"`},
	{"foo/BAR", map[string]string{"bslash": `With \backslash`}, `foo/bar; bslash="With \\backslash"`},
	{"foo/BAR", map[string]string{"both": `With \backslash and "quote`}, `foo/bar; both="With \\backslash and \"quote"`},
	{"foo/BAR", map[string]string{"": "empty attribute"}, ""},
	{"foo/BAR", map[string]string{"bad attribute": "baz"}, ""},
	{"foo/BAR", map[string]string{"nonascii": "not an ascii character: ä"}, ""},
	{"foo/bar", map[string]string{"a": "av", "b": "bv", "c": "cv"}, "foo/bar; a=av; b=bv; c=cv"},
	{"foo/bar", map[string]string{"0": "'", "9": "'"}, "foo/bar; 0='; 9='"},
	{"foo", map[string]string{"bar": ""}, `foo; bar=""`},
}

func TestFormatMediaType(t *testing.T) {
	for i, tt := range formatTests {
		got := FormatMediaType(tt.typ, tt.params)
		if got != tt.want {
			t.Errorf("%d. FormatMediaType(%q, %v) = %q; want %q", i, tt.typ, tt.params, got, tt.want)
		}
	}
}

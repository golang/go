// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mime

import (
	"maps"
	"strings"
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
		{`"C:\dev\go\robots.txt"`, `C:\dev\go\robots.txt`, ""},
		{`"C:\新建文件夹\中文第二次测试.mp4"`, `C:\新建文件夹\中文第二次测试.mp4`, ""},
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

var parseMediaTypeTests []mediaTypeTest

func init() {
	// Convenience map initializer
	m := func(s ...string) map[string]string {
		sm := make(map[string]string)
		for i := 0; i < len(s); i += 2 {
			sm[s[i]] = s[i+1]
		}
		return sm
	}

	nameFoo := map[string]string{"name": "foo"}
	parseMediaTypeTests = []mediaTypeTest{
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

		// #attonly
		{`attachment`,
			"attachment",
			m()},
		// #attonlyucase
		{`ATTACHMENT`,
			"attachment",
			m()},
		// #attwithasciifilename
		{`attachment; filename="foo.html"`,
			"attachment",
			m("filename", "foo.html")},
		// #attwithasciifilename25
		{`attachment; filename="0000000000111111111122222"`,
			"attachment",
			m("filename", "0000000000111111111122222")},
		// #attwithasciifilename35
		{`attachment; filename="00000000001111111111222222222233333"`,
			"attachment",
			m("filename", "00000000001111111111222222222233333")},
		// #attwithasciifnescapedchar
		{`attachment; filename="f\oo.html"`,
			"attachment",
			m("filename", "f\\oo.html")},
		// #attwithasciifnescapedquote
		{`attachment; filename="\"quoting\" tested.html"`,
			"attachment",
			m("filename", `"quoting" tested.html`)},
		// #attwithquotedsemicolon
		{`attachment; filename="Here's a semicolon;.html"`,
			"attachment",
			m("filename", "Here's a semicolon;.html")},
		// #attwithfilenameandextparam
		{`attachment; foo="bar"; filename="foo.html"`,
			"attachment",
			m("foo", "bar", "filename", "foo.html")},
		// #attwithfilenameandextparamescaped
		{`attachment; foo="\"\\";filename="foo.html"`,
			"attachment",
			m("foo", "\"\\", "filename", "foo.html")},
		// #attwithasciifilenameucase
		{`attachment; FILENAME="foo.html"`,
			"attachment",
			m("filename", "foo.html")},
		// #attwithasciifilenamenq
		{`attachment; filename=foo.html`,
			"attachment",
			m("filename", "foo.html")},
		// #attwithasciifilenamenqs
		{`attachment; filename=foo.html ;`,
			"attachment",
			m("filename", "foo.html")},
		// #attwithfntokensq
		{`attachment; filename='foo.html'`,
			"attachment",
			m("filename", "'foo.html'")},
		// #attwithisofnplain
		{`attachment; filename="foo-ä.html"`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attwithutf8fnplain
		{`attachment; filename="foo-Ã¤.html"`,
			"attachment",
			m("filename", "foo-Ã¤.html")},
		// #attwithfnrawpctenca
		{`attachment; filename="foo-%41.html"`,
			"attachment",
			m("filename", "foo-%41.html")},
		// #attwithfnusingpct
		{`attachment; filename="50%.html"`,
			"attachment",
			m("filename", "50%.html")},
		// #attwithfnrawpctencaq
		{`attachment; filename="foo-%\41.html"`,
			"attachment",
			m("filename", "foo-%\\41.html")},
		// #attwithnamepct
		{`attachment; name="foo-%41.html"`,
			"attachment",
			m("name", "foo-%41.html")},
		// #attwithfilenamepctandiso
		{`attachment; name="ä-%41.html"`,
			"attachment",
			m("name", "ä-%41.html")},
		// #attwithfnrawpctenclong
		{`attachment; filename="foo-%c3%a4-%e2%82%ac.html"`,
			"attachment",
			m("filename", "foo-%c3%a4-%e2%82%ac.html")},
		// #attwithasciifilenamews1
		{`attachment; filename ="foo.html"`,
			"attachment",
			m("filename", "foo.html")},
		// #attmissingdisposition
		{`filename=foo.html`,
			"", m()},
		// #attmissingdisposition2
		{`x=y; filename=foo.html`,
			"", m()},
		// #attmissingdisposition3
		{`"foo; filename=bar;baz"; filename=qux`,
			"", m()},
		// #attmissingdisposition4
		{`filename=foo.html, filename=bar.html`,
			"", m()},
		// #emptydisposition
		{`; filename=foo.html`,
			"", m()},
		// #doublecolon
		{`: inline; attachment; filename=foo.html`,
			"", m()},
		// #attandinline
		{`inline; attachment; filename=foo.html`,
			"", m()},
		// #attandinline2
		{`attachment; inline; filename=foo.html`,
			"", m()},
		// #attbrokenquotedfn
		{`attachment; filename="foo.html".txt`,
			"", m()},
		// #attbrokenquotedfn2
		{`attachment; filename="bar`,
			"", m()},
		// #attbrokenquotedfn3
		{`attachment; filename=foo"bar;baz"qux`,
			"", m()},
		// #attmultinstances
		{`attachment; filename=foo.html, attachment; filename=bar.html`,
			"", m()},
		// #attmissingdelim
		{`attachment; foo=foo filename=bar`,
			"", m()},
		// #attmissingdelim2
		{`attachment; filename=bar foo=foo`,
			"", m()},
		// #attmissingdelim3
		{`attachment filename=bar`,
			"", m()},
		// #attreversed
		{`filename=foo.html; attachment`,
			"", m()},
		// #attconfusedparam
		{`attachment; xfilename=foo.html`,
			"attachment",
			m("xfilename", "foo.html")},
		// #attcdate
		{`attachment; creation-date="Wed, 12 Feb 1997 16:29:51 -0500"`,
			"attachment",
			m("creation-date", "Wed, 12 Feb 1997 16:29:51 -0500")},
		// #attmdate
		{`attachment; modification-date="Wed, 12 Feb 1997 16:29:51 -0500"`,
			"attachment",
			m("modification-date", "Wed, 12 Feb 1997 16:29:51 -0500")},
		// #dispext
		{`foobar`, "foobar", m()},
		// #dispextbadfn
		{`attachment; example="filename=example.txt"`,
			"attachment",
			m("example", "filename=example.txt")},
		// #attwithfn2231utf8
		{`attachment; filename*=UTF-8''foo-%c3%a4-%e2%82%ac.html`,
			"attachment",
			m("filename", "foo-ä-€.html")},
		// #attwithfn2231noc
		{`attachment; filename*=''foo-%c3%a4-%e2%82%ac.html`,
			"attachment",
			m()},
		// #attwithfn2231utf8comp
		{`attachment; filename*=UTF-8''foo-a%cc%88.html`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attwithfn2231ws2
		{`attachment; filename*= UTF-8''foo-%c3%a4.html`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attwithfn2231ws3
		{`attachment; filename* =UTF-8''foo-%c3%a4.html`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attwithfn2231quot
		{`attachment; filename*="UTF-8''foo-%c3%a4.html"`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attwithfn2231quot2
		{`attachment; filename*="foo%20bar.html"`,
			"attachment",
			m()},
		// #attwithfn2231singleqmissing
		{`attachment; filename*=UTF-8'foo-%c3%a4.html`,
			"attachment",
			m()},
		// #attwithfn2231nbadpct1
		{`attachment; filename*=UTF-8''foo%`,
			"attachment",
			m()},
		// #attwithfn2231nbadpct2
		{`attachment; filename*=UTF-8''f%oo.html`,
			"attachment",
			m()},
		// #attwithfn2231dpct
		{`attachment; filename*=UTF-8''A-%2541.html`,
			"attachment",
			m("filename", "A-%41.html")},
		// #attfncont
		{`attachment; filename*0="foo."; filename*1="html"`,
			"attachment",
			m("filename", "foo.html")},
		// #attfncontenc
		{`attachment; filename*0*=UTF-8''foo-%c3%a4; filename*1=".html"`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attfncontlz
		{`attachment; filename*0="foo"; filename*01="bar"`,
			"attachment",
			m("filename", "foo")},
		// #attfncontnc
		{`attachment; filename*0="foo"; filename*2="bar"`,
			"attachment",
			m("filename", "foo")},
		// #attfnconts1
		{`attachment; filename*1="foo."; filename*2="html"`,
			"attachment", m()},
		// #attfncontord
		{`attachment; filename*1="bar"; filename*0="foo"`,
			"attachment",
			m("filename", "foobar")},
		// #attfnboth
		{`attachment; filename="foo-ae.html"; filename*=UTF-8''foo-%c3%a4.html`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attfnboth2
		{`attachment; filename*=UTF-8''foo-%c3%a4.html; filename="foo-ae.html"`,
			"attachment",
			m("filename", "foo-ä.html")},
		// #attfnboth3
		{`attachment; filename*0*=ISO-8859-15''euro-sign%3d%a4; filename*=ISO-8859-1''currency-sign%3d%a4`,
			"attachment",
			m()},
		// #attnewandfn
		{`attachment; foobar=x; filename="foo.html"`,
			"attachment",
			m("foobar", "x", "filename", "foo.html")},

		// Browsers also just send UTF-8 directly without RFC 2231,
		// at least when the source page is served with UTF-8.
		{`form-data; firstname="Брэд"; lastname="Фицпатрик"`,
			"form-data",
			m("firstname", "Брэд", "lastname", "Фицпатрик")},

		// Empty string used to be mishandled.
		{`foo; bar=""`, "foo", m("bar", "")},

		// Microsoft browsers in intranet mode do not think they need to escape \ in file name.
		{`form-data; name="file"; filename="C:\dev\go\robots.txt"`, "form-data", m("name", "file", "filename", `C:\dev\go\robots.txt`)},
		{`form-data; name="file"; filename="C:\新建文件夹\中文第二次测试.mp4"`, "form-data", m("name", "file", "filename", `C:\新建文件夹\中文第二次测试.mp4`)},

		// issue #46323 (https://github.com/golang/go/issues/46323)
		{
			// example from rfc2231-p.3 (https://datatracker.ietf.org/doc/html/rfc2231)
			`message/external-body; access-type=URL;
		URL*0="ftp://";
		URL*1="cs.utk.edu/pub/moore/bulk-mailer/bulk-mailer.tar";`, // <-- trailing semicolon
			`message/external-body`,
			m("access-type", "URL", "url", "ftp://cs.utk.edu/pub/moore/bulk-mailer/bulk-mailer.tar"),
		},

		// Issue #48866: duplicate parameters containing equal values should be allowed
		{`text; charset=utf-8; charset=utf-8; format=fixed`, "text", m("charset", "utf-8", "format", "fixed")},
		{`text; charset=utf-8; format=flowed; charset=utf-8`, "text", m("charset", "utf-8", "format", "flowed")},
	}
}

func TestParseMediaType(t *testing.T) {
	for _, test := range parseMediaTypeTests {
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
		if !maps.Equal(params, test.p) {
			t.Errorf("for input %#q, wrong params.\n"+
				"expected: %#v\n"+
				"     got: %#v",
				test.in, test.p, params)
		}
	}
}

func BenchmarkParseMediaType(b *testing.B) {
	for range b.N {
		for _, test := range parseMediaTypeTests {
			ParseMediaType(test.in)
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
	// Tests from http://greenbytes.de/tech/tc2231/
	{`"attachment"`, "attachment", "mime: no media type"},
	{"attachment; filename=foo,bar.html", "attachment", "mime: invalid media parameter"},
	{"attachment; ;filename=foo", "attachment", "mime: invalid media parameter"},
	{"attachment; filename=foo bar.html", "attachment", "mime: invalid media parameter"},
	{`attachment; filename="foo.html"; filename="bar.html"`, "attachment", "mime: duplicate parameter name"},
	{"attachment; filename=foo[1](2).html", "attachment", "mime: invalid media parameter"},
	{"attachment; filename=foo-ä.html", "attachment", "mime: invalid media parameter"},
	{"attachment; filename=foo-Ã¤.html", "attachment", "mime: invalid media parameter"},
	{`attachment; filename *=UTF-8''foo-%c3%a4.html`, "attachment", "mime: invalid media parameter"},
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

func BenchmarkParseMediaTypeBogus(b *testing.B) {
	for range b.N {
		for _, test := range badMediaTypeTests {
			ParseMediaType(test.in)
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
	{"attachment", map[string]string{"filename": "ĄĄŽŽČČŠŠ"}, "attachment; filename*=utf-8''%C4%84%C4%84%C5%BD%C5%BD%C4%8C%C4%8C%C5%A0%C5%A0"},
	{"attachment", map[string]string{"filename": "ÁÁÊÊÇÇÎÎ"}, "attachment; filename*=utf-8''%C3%81%C3%81%C3%8A%C3%8A%C3%87%C3%87%C3%8E%C3%8E"},
	{"attachment", map[string]string{"filename": "数据统计.png"}, "attachment; filename*=utf-8''%E6%95%B0%E6%8D%AE%E7%BB%9F%E8%AE%A1.png"},
	{"foo/BAR", nil, "foo/bar"},
	{"foo/BAR", map[string]string{"X": "Y"}, "foo/bar; x=Y"},
	{"foo/BAR", map[string]string{"space": "With space"}, `foo/bar; space="With space"`},
	{"foo/BAR", map[string]string{"quote": `With "quote`}, `foo/bar; quote="With \"quote"`},
	{"foo/BAR", map[string]string{"bslash": `With \backslash`}, `foo/bar; bslash="With \\backslash"`},
	{"foo/BAR", map[string]string{"both": `With \backslash and "quote`}, `foo/bar; both="With \\backslash and \"quote"`},
	{"foo/BAR", map[string]string{"": "empty attribute"}, ""},
	{"foo/BAR", map[string]string{"bad attribute": "baz"}, ""},
	{"foo/BAR", map[string]string{"nonascii": "not an ascii character: ä"}, "foo/bar; nonascii*=utf-8''not%20an%20ascii%20character%3A%20%C3%A4"},
	{"foo/BAR", map[string]string{"ctl": "newline: \n nil: \000"}, "foo/bar; ctl*=utf-8''newline%3A%20%0A%20nil%3A%20%00"},
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
		if got == "" {
			continue
		}
		typ, params, err := ParseMediaType(got)
		if err != nil {
			t.Errorf("%d. ParseMediaType(%q) err: %v", i, got, err)
		}
		if typ != strings.ToLower(tt.typ) {
			t.Errorf("%d. ParseMediaType(%q) typ = %q; want %q", i, got, typ, tt.typ)
		}
		for k, v := range tt.params {
			k = strings.ToLower(k)
			if params[k] != v {
				t.Errorf("%d. ParseMediaType(%q) params[%s] = %q; want %q", i, got, k, params[k], v)
			}
		}
	}
}
